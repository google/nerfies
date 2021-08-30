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

"""Different model implementation plus a general port for all the models."""
from typing import Any, Dict, Mapping, Optional, Tuple

from flax import linen as nn
import frozendict
from jax import random
import jax.numpy as jnp
from dataclasses import field

from nerfies import configs
from nerfies import glo
from nerfies import model_utils
from nerfies import modules
from nerfies import types
from nerfies import warping


class NerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs.

  Attributes:
    num_coarse_samples: int, the number of samples for coarse nerf.
    num_fine_samples: int, the number of samples for fine nerf.
    use_viewdirs: bool, use viewdirs as a condition.
    near: float, near clip.
    far: float, far clip.
    noise_std: float, std dev of noise added to regularize sigma output.
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_condition_depth: int, the depth of the second part of MLP.
    nerf_condition_width: int, the width of the second part of MLP.
    activation: the activation function used in the MLP.
    sigma_activation: the activation function applied to the sigma density.
    nerf_skips: which layers to add skip layers in the NeRF model.
    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    use_stratified_sampling: use stratified sampling.
    use_white_background: composite rendering on to a white background.
    num_nerf_point_freqs: degree of positional encoding for positions.
    num_nerf_viewdir_freqs: degree of positional encoding for viewdirs.
    use_linear_disparity: sample linearly in disparity rather than depth.

    num_appearance_embeddings: the number of appearance exemplars.
    num_appearance_features: the dimension size for the appearance code.
    num_camera_embeddings: the number of camera exemplars.
    num_camera_features: the dimension size for the camera code.
    num_warp_freqs: the number of frequencies for the warp positional encoding.
    num_warp_embeddings: the number of embeddings for the warp GLO encoder.
    num_warp_features: the number of features for the warp GLO encoder.

    use_appearance_metadata: if True use the appearance metadata.
    use_camera_metadata: if True use the camera metadata.
    use_warp: whether to use the warp field or not.
    use_warp_jacobian: if True the model computes and returns the Jacobian of
      the warped points.
    use_weights: if True return the density weights.
    warp_kwargs: extra keyword arguments for the warp field.
  """
  num_coarse_samples: int
  num_fine_samples: int
  use_viewdirs: bool
  near: float
  far: float
  noise_std: Optional[float]
  nerf_trunk_depth: int
  nerf_trunk_width: int
  nerf_condition_depth: int
  nerf_condition_width: int
  nerf_skips: Tuple[int]
  alpha_channels: int
  rgb_channels: int
  use_stratified_sampling: bool
  num_nerf_point_freqs: int
  num_nerf_viewdir_freqs: int

  num_appearance_embeddings: int
  num_appearance_features: int
  num_camera_embeddings: int
  num_camera_features: int
  num_warp_embeddings: int
  num_warp_features: int
  num_warp_freqs: int

  activation: types.Activation = nn.relu
  sigma_activation: types.Activation = nn.relu
  use_white_background: bool = False
  use_linear_disparity: bool = False
  use_sample_at_infinity: bool = True

  warp_field_type: str = 'se3'
  use_appearance_metadata: bool = False
  use_camera_metadata: bool = False
  use_warp: bool = False
  use_warp_jacobian: bool = False
  use_weights: bool = False
  warp_kwargs: Mapping[str, Any] = field(default_factory=frozendict.frozendict)

  metadata_encoded: bool = False

  def setup(self):
    self.point_encoder = model_utils.vmap_module(
        modules.SinusoidalEncoder, num_batch_dims=2)(
            num_freqs=self.num_nerf_point_freqs)
    self.viewdir_encoder = model_utils.vmap_module(
        modules.SinusoidalEncoder, num_batch_dims=1)(
            num_freqs=self.num_nerf_viewdir_freqs)
    if self.use_appearance_metadata:
      self.appearance_encoder = glo.GloEncoder(
          num_embeddings=self.num_appearance_embeddings,
          features=self.num_appearance_features)
    if self.use_camera_metadata:
      self.camera_encoder = glo.GloEncoder(
          num_embeddings=self.num_camera_embeddings,
          features=self.num_camera_features)

    self.nerf_coarse = modules.NerfMLP(
        nerf_trunk_depth=self.nerf_trunk_depth,
        nerf_trunk_width=self.nerf_trunk_width,
        nerf_condition_depth=self.nerf_condition_depth,
        nerf_condition_width=self.nerf_condition_width,
        activation=self.activation,
        skips=self.nerf_skips,
        alpha_channels=self.alpha_channels,
        rgb_channels=self.rgb_channels)
    if self.num_fine_samples > 0:
      self.nerf_fine = modules.NerfMLP(
          nerf_trunk_depth=self.nerf_trunk_depth,
          nerf_trunk_width=self.nerf_trunk_width,
          nerf_condition_depth=self.nerf_condition_depth,
          nerf_condition_width=self.nerf_condition_width,
          activation=self.activation,
          skips=self.nerf_skips,
          alpha_channels=self.alpha_channels,
          rgb_channels=self.rgb_channels)
    else:
      self.nerf_fine = None

    if self.use_warp:
      self.warp_field = warping.create_warp_field(
          field_type=self.warp_field_type,
          num_freqs=self.num_warp_freqs,
          num_embeddings=self.num_warp_embeddings,
          num_features=self.num_warp_features,
          num_batch_dims=2,
          **self.warp_kwargs)

  def __call__(
      self,
      rays_dict: Dict[str, Any],
      warp_alpha: float = None,
      metadata_encoded=False,
      use_warp=True,
      deterministic=False,
  ):
    """Nerf Model.

    Args:
      rays_dict: a dictionary containing the ray information. Contains:
        'origins': the ray origins.
        'directions': unit vectors which are the ray directions.
        'viewdirs': (optional) unit vectors which are viewing directions.
        'metadata': a dictionary of metadata indices e.g., for warping.
      warp_alpha: the alpha for the positional encoding.
      metadata_encoded: if True, assume the metadata is already encoded.
      use_warp: if True use the warp field (if also enabled in the model).
      deterministic: whether evaluation should be deterministic.

    Returns:
      ret: list, [(rgb, disp, acc), (rgb_coarse, disp_coarse, acc_coarse)]
    """
    # Extract viewdirs from the ray array
    origins = rays_dict['origins']
    directions = rays_dict['directions']
    metadata = rays_dict['metadata']
    if 'viewdirs' in rays_dict:
      viewdirs = rays_dict['viewdirs']
    else:  # viewdirs are normalized rays_d
      viewdirs = directions
    # Stratified sampling along rays
    z_vals, points = model_utils.sample_along_rays(
        self.make_rng('coarse'), origins, directions, self.num_coarse_samples,
        self.near, self.far, self.use_stratified_sampling,
        self.use_linear_disparity)
    if self.use_warp and use_warp:
      metadata_channels = self.num_warp_features if metadata_encoded else 1
      warp_metadata = jnp.broadcast_to(
          metadata['warp'][:, jnp.newaxis, :],
          shape=(*points.shape[:2], metadata_channels))
      warp_ret = self.warp_field(
          points, warp_metadata, warp_alpha, self.use_warp_jacobian,
          metadata_encoded)
      if self.use_warp_jacobian:
        points, warp_jacobian = warp_ret
      else:
        points = warp_ret

    points_embed = self.point_encoder(points)

    condition_inputs = []

    # Point attribute predictions
    if self.use_viewdirs:
      viewdirs_embed = self.viewdir_encoder(viewdirs)
      condition_inputs.append(viewdirs_embed)

    if self.use_appearance_metadata:
      if metadata_encoded:
        appearance_code = metadata['appearance']
      else:
        appearance_code = self.appearance_encoder(metadata['appearance'])
      condition_inputs.append(appearance_code)
    if self.use_camera_metadata:
      if metadata_encoded:
        camera_code = metadata['camera']
      else:
        camera_code = self.camera_encoder(metadata['camera'])
      condition_inputs.append(camera_code)

    # The condition inputs have a shape of (B, C) now rather than (B, S, C)
    # since we assume all samples have the same condition input. We might want
    # to change this later.
    condition_inputs = (
        jnp.concatenate(condition_inputs, axis=-1)
        if condition_inputs else None)

    coarse_raw = self.nerf_coarse(points_embed, condition=condition_inputs)
    # Add noises to regularize the density predictions if needed
    coarse_raw = model_utils.noise_regularize(
        self.make_rng('coarse'), coarse_raw, self.noise_std,
        self.use_stratified_sampling)
    # Volumetric rendering.
    rgb, exp_depth, med_depth, disp, acc, weights = (
        model_utils.volumetric_rendering(
            coarse_raw,
            z_vals,
            directions,
            use_white_background=self.use_white_background,
            sigma_activation=self.sigma_activation,
            sample_at_infinity=self.use_sample_at_infinity))

    out = {
        'coarse': {
            'rgb': rgb,
            'depth': exp_depth,
            'med_depth': med_depth,
            'disp': disp,
            'acc': acc,
        }
    }
    if self.use_warp and use_warp and self.use_warp_jacobian:
      out['coarse']['warp_jacobian'] = warp_jacobian
    if self.use_weights:
      out['coarse']['weights'] = weights
    # Hierarchical sampling based on coarse predictions
    if self.num_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
      z_vals, points = model_utils.sample_pdf(
          self.make_rng('fine'),
          z_vals_mid,
          weights[..., 1:-1],
          origins,
          directions,
          z_vals,
          self.num_fine_samples,
          self.use_stratified_sampling,
      )
      if self.use_warp and use_warp:
        metadata_channels = self.num_warp_features if metadata_encoded else 1
        warp_metadata = jnp.broadcast_to(
            metadata['warp'][:, jnp.newaxis, :],
            shape=(*points.shape[:2], metadata_channels))
        points = self.warp_field(
            points, warp_metadata, warp_alpha, False, metadata_encoded)
      points_embed = self.point_encoder(points)
      fine_raw = self.nerf_fine(points_embed, condition=condition_inputs)
      fine_raw = model_utils.noise_regularize(
          self.make_rng('fine'), fine_raw, self.noise_std,
          self.use_stratified_sampling)
      rgb, exp_depth, med_depth, disp, acc, weights = (
          model_utils.volumetric_rendering(
              fine_raw,
              z_vals,
              directions,
              use_white_background=self.use_white_background,
              sigma_activation=self.sigma_activation,
              sample_at_infinity=self.use_sample_at_infinity))
      out['fine'] = {
          'rgb': rgb,
          'depth': exp_depth,
          'med_depth': med_depth,
          'disp': disp,
          'acc': acc,
      }
      if self.use_weights:
        out['fine']['weights'] = weights

    return out


def nerf(key,
         config: configs.ModelConfig,
         batch_size: int,
         num_appearance_embeddings: int,
         num_camera_embeddings: int,
         num_warp_embeddings: int,
         near: float,
         far: float,
         use_warp_jacobian: bool = False,
         use_weights: bool = False):
  """Neural Randiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    config: model configs.
    batch_size: the evaluation batch size used for shape inference.
    num_appearance_embeddings: the number of appearance embeddings.
    num_camera_embeddings: the number of camera embeddings.
    num_warp_embeddings: the number of warp embeddings.
    near: the near plane of the scene.
    far: the far plane of the scene.
    use_warp_jacobian: if True the model computes and returns the Jacobian of
      the warped points.
    use_weights: if True return the density weights from the NeRF.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  num_nerf_point_freqs = config.num_nerf_point_freqs
  num_nerf_viewdir_freqs = config.num_nerf_viewdir_freqs
  num_coarse_samples = config.num_coarse_samples
  num_fine_samples = config.num_fine_samples
  use_viewdirs = config.use_viewdirs
  noise_std = config.noise_std
  use_stratified_sampling = config.use_stratified_sampling
  use_white_background = config.use_white_background
  nerf_trunk_depth = config.nerf_trunk_depth
  nerf_trunk_width = config.nerf_trunk_width
  nerf_condition_depth = config.nerf_condition_depth
  nerf_condition_width = config.nerf_condition_width
  nerf_skips = config.nerf_skips
  alpha_channels = config.alpha_channels
  rgb_channels = config.rgb_channels
  use_linear_disparity = config.use_linear_disparity

  model = NerfModel(
      num_coarse_samples=num_coarse_samples,
      num_fine_samples=num_fine_samples,
      use_viewdirs=use_viewdirs,
      near=near,
      far=far,
      noise_std=noise_std,
      nerf_trunk_depth=nerf_trunk_depth,
      nerf_trunk_width=nerf_trunk_width,
      nerf_condition_depth=nerf_condition_depth,
      nerf_condition_width=nerf_condition_width,
      activation=config.activation,
      sigma_activation=config.sigma_activation,
      nerf_skips=nerf_skips,
      alpha_channels=alpha_channels,
      rgb_channels=rgb_channels,
      use_stratified_sampling=use_stratified_sampling,
      use_white_background=use_white_background,
      use_sample_at_infinity=config.use_sample_at_infinity,
      num_nerf_point_freqs=num_nerf_point_freqs,
      num_nerf_viewdir_freqs=num_nerf_viewdir_freqs,
      use_linear_disparity=use_linear_disparity,
      use_warp_jacobian=use_warp_jacobian,
      use_weights=use_weights,
      use_appearance_metadata=config.use_appearance_metadata,
      use_camera_metadata=config.use_camera_metadata,
      use_warp=config.use_warp,
      num_appearance_embeddings=num_appearance_embeddings,
      num_appearance_features=config.appearance_metadata_dims,
      num_camera_embeddings=num_camera_embeddings,
      num_camera_features=config.camera_metadata_dims,
      num_warp_embeddings=num_warp_embeddings,
      num_warp_freqs=config.num_warp_freqs,
      num_warp_features=config.num_warp_features,
      warp_field_type=config.warp_field_type,
      warp_kwargs=config.warp_kwargs,
  )

  init_rays_dict = {
      'origins': jnp.ones((batch_size, 3), jnp.float32),
      'directions': jnp.ones((batch_size, 3), jnp.float32),
      'metadata': {
          'warp': jnp.ones((batch_size, 1), jnp.uint32),
          'camera': jnp.ones((batch_size, 1), jnp.uint32),
          'appearance': jnp.ones((batch_size, 1), jnp.uint32),
      }
  }

  key, key1, key2 = random.split(key, 3)
  params = model.init({
      'params': key,
      'coarse': key1,
      'fine': key2
  }, init_rays_dict, warp_alpha=0.0)['params']

  return model, params
