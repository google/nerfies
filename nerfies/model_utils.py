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

"""Helper functions/classes for model definition."""

from flax import linen as nn
from flax import optim
from flax import struct
from jax import lax
from jax import random
import jax.numpy as jnp


@struct.dataclass
class TrainState:
  optimizer: optim.Optimizer
  warp_alpha: jnp.ndarray = 0.0
  time_alpha: jnp.ndarray = 0.0

  @property
  def warp_extra(self):
    return {'alpha': self.warp_alpha, 'time_alpha': self.time_alpha}


def sample_along_rays(key, origins, directions, num_coarse_samples, near, far,
                      use_stratified_sampling, use_linear_disparity):
  """Stratified sampling along the rays.

  Args:
    key: jnp.ndarray, random generator key.
    origins: ray origins.
    directions: ray directions.
    num_coarse_samples: int.
    near: float, near clip.
    far: float, far clip.
    use_stratified_sampling: use stratified sampling.
    use_linear_disparity: sampling linearly in disparity rather than depth.

  Returns:
    z_vals: jnp.ndarray, [batch_size, num_coarse_samples], sampled z values.
    points: jnp.ndarray, [batch_size, num_coarse_samples, 3], sampled points.
  """
  batch_size = origins.shape[0]

  t_vals = jnp.linspace(0., 1., num_coarse_samples)
  if not use_linear_disparity:
    z_vals = near * (1. - t_vals) + far * t_vals
  else:
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
  if use_stratified_sampling:
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = jnp.concatenate([mids, z_vals[..., -1:]], -1)
    lower = jnp.concatenate([z_vals[..., :1], mids], -1)
    t_rand = random.uniform(key, [batch_size, num_coarse_samples])
    z_vals = lower + (upper - lower) * t_rand
  else:
    # Broadcast z_vals to make the returned shape consistent.
    z_vals = jnp.broadcast_to(z_vals[None, ...],
                              [batch_size, num_coarse_samples])

  return (z_vals, (origins[..., None, :] +
                   z_vals[..., :, None] * directions[..., None, :]))


def volumetric_rendering(rgb,
                         sigma,
                         z_vals,
                         dirs,
                         use_white_background,
                         sample_at_infinity=True,
                         return_weights=False,
                         eps=1e-10):
  """Volumetric Rendering Function.

  Args:
    rgb: an array of size (B,S,3) containing the RGB color values.
    sigma: an array of size (B,S,1) containing the densities.
    z_vals: an array of size (B,S) containing the z-coordinate of the samples.
    dirs: an array of size (B,3) containing the directions of rays.
    use_white_background: whether to assume a white background or not.
    sample_at_infinity: if True adds a sample at infinity.
    return_weights: if True returns the weights in the dictionary.
    eps: a small number to prevent numerical issues.

  Returns:
    A dictionary containing:
      rgb: an array of size (B,3) containing the rendered colors.
      depth: an array of size (B,) containing the rendered depth.
      acc: an array of size (B,) containing the accumulated density.
      weights: an array of size (B,S) containing the weight of each sample.
  """
  # TODO(keunhong): remove this hack.
  last_sample_z = 1e10 if sample_at_infinity else 1e-19
  dists = jnp.concatenate([
      z_vals[..., 1:] - z_vals[..., :-1],
      jnp.broadcast_to(last_sample_z, z_vals[..., :1].shape)
  ], -1)
  dists = dists * jnp.linalg.norm(dirs[..., None, :], axis=-1)
  alpha = 1.0 - jnp.exp(-sigma * dists)
  # Prepend a 1.0 to make this an 'exclusive' cumprod as in `tf.math.cumprod`.
  accum_prod = jnp.concatenate([
      jnp.ones_like(alpha[..., :1], alpha.dtype),
      jnp.cumprod(1.0 - alpha[..., :-1] + eps, axis=-1),
  ], axis=-1)
  weights = alpha * accum_prod

  rgb = (weights[..., None] * rgb).sum(axis=-2)
  exp_depth = (weights * z_vals).sum(axis=-1)
  med_depth = compute_depth_map(weights, z_vals)
  acc = weights.sum(axis=-1)
  if use_white_background:
    rgb = rgb + (1. - acc[..., None])

  if sample_at_infinity:
    acc = weights[..., :-1].sum(axis=-1)

  out = {
      'rgb': rgb,
      'depth': exp_depth,
      'med_depth': med_depth,
      'acc': acc,
  }
  if return_weights:
    out['weights'] = weights
  return out


def piecewise_constant_pdf(key, bins, weights, num_coarse_samples,
                           use_stratified_sampling):
  """Piecewise-Constant PDF sampling.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, n_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, n_bins].
    num_coarse_samples: int, the number of samples.
    use_stratified_sampling: bool, use use_stratified_sampling samples.

  Returns:
    z_samples: jnp.ndarray(float32), [batch_size, num_coarse_samples].
  """
  eps = 1e-5

  # Get pdf
  weights += eps  # prevent nans
  pdf = weights / weights.sum(axis=-1, keepdims=True)
  cdf = jnp.cumsum(pdf, axis=-1)
  cdf = jnp.concatenate([jnp.zeros(list(cdf.shape[:-1]) + [1]), cdf], axis=-1)

  # Take uniform samples
  if use_stratified_sampling:
    u = random.uniform(key, list(cdf.shape[:-1]) + [num_coarse_samples])
  else:
    u = jnp.linspace(0., 1., num_coarse_samples)
    u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_coarse_samples])

  # Invert CDF. This takes advantage of the fact that `bins` is sorted.
  mask = (u[..., None, :] >= cdf[..., :, None])

  def minmax(x):
    x0 = jnp.max(jnp.where(mask, x[..., None], x[..., :1, None]), -2)
    x1 = jnp.min(jnp.where(~mask, x[..., None], x[..., -1:, None]), -2)
    x0 = jnp.minimum(x0, x[..., -2:-1])
    x1 = jnp.maximum(x1, x[..., 1:2])
    return x0, x1

  bins_g0, bins_g1 = minmax(bins)
  cdf_g0, cdf_g1 = minmax(cdf)

  denom = (cdf_g1 - cdf_g0)
  denom = jnp.where(denom < eps, 1., denom)
  t = (u - cdf_g0) / denom
  z_samples = bins_g0 + t * (bins_g1 - bins_g0)

  # Prevent gradient from backprop-ing through samples
  return lax.stop_gradient(z_samples)


def sample_pdf(key, bins, weights, origins, directions, z_vals,
               num_coarse_samples, use_stratified_sampling):
  """Hierarchical sampling.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, n_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, n_bins].
    origins: ray origins.
    directions: ray directions.
    z_vals: jnp.ndarray(float32), [batch_size, n_coarse_samples].
    num_coarse_samples: int, the number of samples.
    use_stratified_sampling: bool, use use_stratified_sampling samples.

  Returns:
    z_vals: jnp.ndarray(float32),
      [batch_size, n_coarse_samples + num_fine_samples].
    points: jnp.ndarray(float32),
      [batch_size, n_coarse_samples + num_fine_samples, 3].
  """
  z_samples = piecewise_constant_pdf(key, bins, weights, num_coarse_samples,
                                     use_stratified_sampling)
  # Compute united z_vals and sample points
  z_vals = jnp.sort(jnp.concatenate([z_vals, z_samples], axis=-1), axis=-1)
  return z_vals, (
      origins[..., None, :] + z_vals[..., None] * directions[..., None, :])


def compute_opaqueness_mask(weights, depth_threshold=0.5):
  """Computes a mask which will be 1.0 at the depth point.

  Args:
    weights: the density weights from NeRF.
    depth_threshold: the accumulation threshold which will be used as the depth
      termination point.

  Returns:
    A tensor containing a mask with the same size as weights that has one
      element long the sample dimension that is 1.0. This element is the point
      where the 'surface' is.
  """
  cumulative_contribution = jnp.cumsum(weights, axis=-1)
  depth_threshold = jnp.array(depth_threshold, dtype=weights.dtype)
  opaqueness = cumulative_contribution >= depth_threshold
  false_padding = jnp.zeros_like(opaqueness[..., :1])
  padded_opaqueness = jnp.concatenate(
      [false_padding, opaqueness[..., :-1]], axis=-1)
  opaqueness_mask = jnp.logical_xor(opaqueness, padded_opaqueness)
  opaqueness_mask = opaqueness_mask.astype(weights.dtype)
  return opaqueness_mask


def compute_depth_index(weights, depth_threshold=0.5):
  """Compute the sample index of the median depth accumulation."""
  opaqueness_mask = compute_opaqueness_mask(weights, depth_threshold)
  return jnp.argmax(opaqueness_mask, axis=-1)


def compute_depth_map(weights, z_vals, depth_threshold=0.5):
  """Compute the depth using the median accumulation.

  Note that this differs from the depth computation in NeRF-W's codebase!

  Args:
    weights: the density weights from NeRF.
    z_vals: the z coordinates of the samples.
    depth_threshold: the accumulation threshold which will be used as the depth
      termination point.

  Returns:
    A tensor containing the depth of each input pixel.
  """
  opaqueness_mask = compute_opaqueness_mask(weights, depth_threshold)
  return jnp.sum(opaqueness_mask * z_vals, axis=-1)


def noise_regularize(key, raw, noise_std, use_stratified_sampling):
  """Regularize the density prediction by adding gaussian noise.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    raw: jnp.ndarray(float32), [batch_size, num_coarse_samples, 4].
    noise_std: float, std dev of noise added to regularize sigma output.
    use_stratified_sampling: add noise only if use_stratified_sampling is True.

  Returns:
    raw: jnp.ndarray(float32), [batch_size, num_coarse_samples, 4], updated raw.
  """
  if (noise_std is not None) and noise_std > 0.0 and use_stratified_sampling:
    unused_key, key = random.split(key)
    noise = random.normal(key, raw[..., 3:4].shape, dtype=raw.dtype) * noise_std
    raw = jnp.concatenate([raw[..., :3], raw[..., 3:4] + noise], axis=-1)
  return raw


def broadcast_feature_to(array: jnp.ndarray, shape: jnp.shape):
  """Matches the shape dimensions (everything except the channel dims).

  This is useful when you watch to match the shape of two features that have
  a different number of channels.

  Args:
    array: the array to broadcast.
    shape: the shape to broadcast the tensor to.

  Returns:
    The broadcasted tensor.
  """
  out_shape = (*shape[:-1], array.shape[-1])
  return jnp.broadcast_to(array, out_shape)


def metadata_like(rays, metadata_id):
  """Create a metadata array like a ray batch."""
  return jnp.full_like(rays[..., :1], fill_value=metadata_id, dtype=jnp.uint32)


def vmap_module(module, in_axes=0, out_axes=0, num_batch_dims=1):
  """Vectorize a module.

  Args:
    module: the module to vectorize.
    in_axes: the `in_axes` argument passed to vmap. See `jax.vmap`.
    out_axes: the `out_axes` argument passed to vmap. See `jax.vmap`.
    num_batch_dims: the number of batch dimensions (how many times to apply vmap
      to the module).

  Returns:
    A vectorized module.
  """
  for _ in range(num_batch_dims):
    module = nn.vmap(
        module,
        variable_axes={'params': None},
        split_rngs={'params': False},
        in_axes=in_axes,
        out_axes=out_axes)

  return module


def identity_initializer(_, shape):
  max_shape = max(shape)
  return jnp.eye(max_shape)[:shape[0], :shape[1]]
