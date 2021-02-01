# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library to training NeRFs."""
import functools
from typing import Any
from typing import Callable
from typing import Dict

from absl import logging
from flax import struct
from flax.training import checkpoints
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import vmap

from nerfies import model_utils
from nerfies import models
from nerfies import utils
from nerfies import warping


@struct.dataclass
class ScalarParams:
  learning_rate: float
  elastic_loss_weight: float = 0.0
  background_loss_weight: float = 0.0
  background_noise_std: float = 0.001


def save_checkpoint(path, state, keep=100):
  """Save the state to a checkpoint."""
  state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
  step = state_to_save.optimizer.state.step
  checkpoint_path = checkpoints.save_checkpoint(
      path, state_to_save, step, keep=keep)
  logging.info('Saved checkpoint: step=%d, path=%s', int(step), checkpoint_path)
  return checkpoint_path


@jax.jit
def compute_elastic_loss(jacobian, alpha=-2.0, scale=0.03, eps=1e-6):
  """Compute the elastic regularization loss.

  The loss is given by sum(log(S)^2). This penalizes the singular values
  when they deviate from the identity since log(1) = 0.0,
  where D is the diagonal matrix containing the singular values.

  Args:
    jacobian: the Jacobian of the point transformation.
    alpha: the alpha for the General loss.
    scale: the scale for the General loss.
    eps: a small value to prevent taking the log of zero.

  Returns:
    The elastic regularization loss.
  """
  svals = jnp.linalg.svd(jacobian, compute_uv=False)
  log_svals = jnp.log(jnp.maximum(svals, eps))
  sq_residual = jnp.sum(log_svals**2, axis=-1)
  loss = scale * utils.general_loss_with_squared_residual(
      sq_residual, alpha=alpha, scale=scale)
  residual = jnp.sqrt(sq_residual)
  return loss, residual


@functools.partial(jax.jit, static_argnums=0)
def compute_background_loss(
    model, state, params, key, points, noise_std, alpha=-2, scale=0.03):
  """Compute the background regularization loss."""
  metadata = random.randint(key,
                            (points.shape[0], 1),
                            minval=0,
                            maxval=model.num_warp_embeddings,
                            dtype=jnp.uint32)
  point_noise = noise_std * random.normal(key, points.shape)
  points = points + point_noise
  warp_field = warping.create_warp_field(
      field_type=model.warp_field_type,
      num_freqs=model.num_warp_freqs,
      num_embeddings=model.num_warp_embeddings,
      num_features=model.num_warp_features,
      num_batch_dims=1,
      **model.warp_kwargs)
  warped_points = warp_field.apply(
      {'params': params['warp_field']},
      points, metadata, state.warp_alpha, False, False)
  sq_residual = jnp.sum((warped_points - points)**2, axis=-1)
  loss = scale * utils.general_loss_with_squared_residual(
      sq_residual, alpha=alpha, scale=scale)
  return loss


def train_step(model: models.NerfModel,
               rng_key: Callable[[int], jnp.ndarray],
               state,
               batch: Dict[str, Any],
               scalar_params: ScalarParams,
               use_elastic_loss: bool = False,
               elastic_reduce_method: str = 'median',
               use_background_loss: bool = False):
  """One optimization step.

  Args:
    model: the model module to evaluate.
    rng_key: The random number generator.
    state: model_utils.TrainState, state of model and optimizer.
    batch: dict. A mini-batch of data for training.
    scalar_params: scalar-valued parameters.
    use_elastic_loss: is True use the elastic regularization loss.
    elastic_reduce_method: which method to use to reduce the samples for the
      elastic loss. 'median' selects the median depth point sample while
      'weight' computes a weighted sum using the density weights.
    use_background_loss: if True use the background regularization loss.

  Returns:
    new_state: model_utils.TrainState, new training state.
    stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
  """
  rng_key, fine_key, coarse_key, reg_key = random.split(rng_key, 4)

  # pylint: disable=unused-argument
  def _compute_loss_and_stats(params, model_out, use_elastic_loss=False):
    rgb_loss = ((model_out['rgb'] - batch['rgb'][..., :3])**2).mean()
    stats = {
        'loss/rgb': rgb_loss,
    }
    loss = rgb_loss
    if use_elastic_loss:
      v_elastic_fn = jax.jit(vmap(vmap(compute_elastic_loss)))
      weights = lax.stop_gradient(model_out['weights'])
      jacobian = model_out['warp_jacobian']
      # Pick the median point Jacobian.
      if elastic_reduce_method == 'median':
        depth_indices = model_utils.compute_depth_index(weights)
        jacobian = jnp.take_along_axis(
            # Unsqueeze axes: sample axis, Jacobian row, Jacobian col.
            jacobian, depth_indices[..., None, None, None], axis=-3)
      # Compute loss using Jacobian.
      elastic_loss, elastic_residual = v_elastic_fn(jacobian)
      # Multiply weight if weighting by density.
      if elastic_reduce_method == 'weight':
        elastic_loss = weights * elastic_loss
      elastic_loss = elastic_loss.sum(axis=-1).mean()
      stats['loss/elastic'] = elastic_loss
      stats['residual/elastic'] = jnp.mean(elastic_residual)
      loss += scalar_params.elastic_loss_weight * elastic_loss

    if 'warp_jacobian' in model_out:
      jacobian = model_out['warp_jacobian']
      jacobian_det = jnp.linalg.det(jacobian)
      jacobian_div = utils.jacobian_to_div(jacobian)
      jacobian_curl = utils.jacobian_to_curl(jacobian)
      stats['metric/jacobian_det'] = jnp.mean(jacobian_det)
      stats['metric/jacobian_div'] = jnp.mean(jacobian_div)
      stats['metric/jacobian_curl'] = jnp.mean(
          jnp.linalg.norm(jacobian_curl, axis=-1))

    stats['loss/total'] = loss
    stats['metric/psnr'] = utils.compute_psnr(rgb_loss)
    return loss, stats

  def _loss_fn(params):
    ret = model.apply({'params': params['model']},
                      batch,
                      warp_alpha=state.warp_alpha,
                      rngs={
                          'fine': fine_key,
                          'coarse': coarse_key
                      })

    losses = {}
    stats = {}
    if 'fine' in ret:
      losses['fine'], stats['fine'] = _compute_loss_and_stats(
          params, ret['fine'])
    if 'coarse' in ret:
      losses['coarse'], stats['coarse'] = _compute_loss_and_stats(
          params, ret['coarse'], use_elastic_loss=use_elastic_loss)

    if use_background_loss:
      background_loss = compute_background_loss(
          model,
          state=state,
          params=params['model'],
          key=reg_key,
          points=batch['background_points'],
          noise_std=scalar_params.background_noise_std)
      background_loss = background_loss.mean()
      losses['background'] = (
          scalar_params.background_loss_weight * background_loss)
      stats['background_loss'] = background_loss

    return sum(losses.values()), stats

  optimizer = state.optimizer
  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (_, stats), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, axis_name='batch')
  stats = jax.lax.pmean(stats, axis_name='batch')
  new_optimizer = optimizer.apply_gradient(
      grad, learning_rate=scalar_params.learning_rate)
  new_state = state.replace(optimizer=new_optimizer)
  return new_state, stats, rng_key
