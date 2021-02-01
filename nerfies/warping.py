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

"""Warp fields."""
from typing import Iterable, Optional, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp

from nerfies import glo
from nerfies import model_utils
from nerfies import modules
from nerfies import quaternion
from nerfies import types


def create_warp_field(
    field_type: str,
    num_freqs: int,
    num_embeddings: int,
    num_features: int,
    num_batch_dims: int,
    **kwargs):
  """Factory function for warp fields."""
  if field_type == 'translation':
    warp_field_cls = TranslationField
  elif field_type == 'se3':
    warp_field_cls = SE3Field
  else:
    raise ValueError(f'Unknown warp field type: {field_type!r}')

  v_warp_field_cls = model_utils.vmap_module(
      warp_field_cls,
      num_batch_dims=num_batch_dims,
      # (points, metadata, alpha, return_jacobian, metadata_encoded).
      in_axes=(0, 0, None, None, None))

  return v_warp_field_cls(
      num_freqs=num_freqs,
      num_embeddings=num_embeddings,
      num_embedding_features=num_features,
      **kwargs)


class MLP(nn.Module):
  """Basic MLP class with hidden layers and an output layers."""
  depth: int
  width: int
  hidden_init: types.Initializer
  hidden_activation: types.Activation = nn.relu
  output_init: Optional[types.Initializer] = None
  output_channels: int = 0
  output_activation: Optional[types.Activation] = lambda x: x
  use_bias: bool = True
  skips: Tuple[int] = tuple()

  @nn.compact
  def __call__(self, x):
    inputs = x
    for i in range(self.depth):
      layer = nn.Dense(
          self.width,
          use_bias=self.use_bias,
          kernel_init=self.hidden_init,
          name=f'hidden_{i}')
      if i in self.skips:
        x = jnp.concatenate([x, inputs], axis=-1)
      x = layer(x)
      x = self.hidden_activation(x)

    if self.output_channels > 0:
      logit_layer = nn.Dense(
          self.output_channels,
          use_bias=self.use_bias,
          kernel_init=self.output_init,
          name='logit')
      x = logit_layer(x)
      if self.output_activation is not None:
        x = self.output_activation(x)

    return x


class TranslationField(nn.Module):
  """Network that predicts warps as a translation field.

  References:
    https://en.wikipedia.org/wiki/Vector_potential
    https://en.wikipedia.org/wiki/Helmholtz_decomposition

  Attributes:
    points_encoder: the positional encoder for the points.
    metadata_encoder: an encoder for metadata.
    alpha: the alpha for the positional encoding.
    skips: the index of the layers with skip connections.
    depth: the depth of the network excluding the output layer.
    hidden_channels: the width of the network hidden layers.
    activation: the activation for each layer.
    metadata_encoded: whether the metadata parameter is pre-encoded or not.
    hidden_initializer: the initializer for the hidden layers.
    output_initializer: the initializer for the last output layer.
  """
  num_freqs: int
  num_embeddings: int
  num_embedding_features: int
  max_freq_log2: Optional[int] = None

  skips: Iterable[int] = (4,)
  depth: int = 6
  hidden_channels: int = 128
  activation: types.Activation = nn.relu
  hidden_init: types.Initializer = nn.initializers.xavier_uniform()
  output_init: types.Initializer = nn.initializers.uniform(scale=1e-4)

  def setup(self):
    self.points_encoder = modules.AnnealedSinusoidalEncoder(
        num_freqs=self.num_freqs, max_freq_log2=self.max_freq_log2)
    self.metadata_encoder = glo.GloEncoder(
        num_embeddings=self.num_embeddings,
        features=self.num_embedding_features)
    # Note that this must be done this way instead of using mutable list
    # operations.
    # See https://github.com/google/flax/issues/524.
    # pylint: disable=g-complex-comprehension
    output_dims = 3
    self.mlp = MLP(
        width=self.hidden_channels,
        depth=self.depth,
        skips=self.skips,
        hidden_init=self.hidden_init,
        output_init=self.output_init,
        output_channels=output_dims)

  def warp(self,
           points: jnp.ndarray,
           metadata: jnp.ndarray,
           alpha: Optional[float] = None,
           metadata_encoded: bool = False):
    points_embed = self.points_encoder(points, alpha=alpha)
    metadata_embed = (metadata
                      if metadata_encoded
                      else self.metadata_encoder(metadata))
    inputs = jnp.concatenate([points_embed, metadata_embed], axis=-1)
    translation = self.mlp(inputs)
    warped_points = points + translation

    return warped_points

  def __call__(self,
               points: jnp.ndarray,
               metadata: jnp.ndarray,
               alpha: Optional[float] = None,
               return_jacobian: bool = False,
               metadata_encoded: bool = False):
    """Warp the given points using a warp field.

    Args:
      points: the points to warp.
      metadata: metadata indices if metadata_encoded is False else pre-encoded
        metadata.
      alpha: the alpha value for the positional encoding.
      return_jacobian: if True compute and return the Jacobian of the warp.
      metadata_encoded: if True assumes the metadata is already encoded.

    Returns:
      The warped points and the Jacobian of the warp if `return_jacobian` is
        True.
    """
    warped_points = self.warp(
        points, metadata, alpha, metadata_encoded)

    if return_jacobian:
      jac_fn = jax.jacfwd(self.warp, argnums=0)
      jac = jac_fn(points, metadata, alpha, metadata_encoded)
      return warped_points, jac

    return warped_points


class SE3Field(nn.Module):
  """Network that predicts warps as an SE(3) field.

  Attributes:
    points_encoder: the positional encoder for the points.
    metadata_encoder: an encoder for metadata.
    alpha: the alpha for the positional encoding.
    skips: the index of the layers with skip connections.
    depth: the depth of the network excluding the logit layer.
    hidden_channels: the width of the network hidden layers.
    activation: the activation for each layer.
    metadata_encoded: whether the metadata parameter is pre-encoded or not.
    hidden_initializer: the initializer for the hidden layers.
    output_initializer: the initializer for the last logit layer.
  """
  num_freqs: int
  num_embeddings: int
  num_embedding_features: int
  max_freq_log2: Optional[int] = None

  activation: types.Activation = nn.relu
  skips: Iterable[int] = (4,)
  trunk_depth: int = 5
  trunk_width: int = 128
  rotation_depth: int = 1
  rotation_width: int = 128
  pivot_depth: int = 1
  pivot_width: int = 128
  translation_depth: int = 1
  translation_width: int = 128

  default_init: types.Initializer = nn.initializers.xavier_uniform()
  rotation_init: types.Initializer = nn.initializers.uniform(scale=1e-4)
  pivot_init: types.Initializer = nn.initializers.uniform(scale=1e-4)
  translation_init: types.Initializer = nn.initializers.uniform(scale=1e-4)

  use_pivot: bool = True
  use_translation: bool = True

  def setup(self):
    self.points_encoder = modules.AnnealedSinusoidalEncoder(
        num_freqs=self.num_freqs, max_freq_log2=self.max_freq_log2)
    self.metadata_encoder = glo.GloEncoder(
        num_embeddings=self.num_embeddings,
        features=self.num_embedding_features)
    self.trunk = MLP(
        depth=self.trunk_depth,
        width=self.trunk_width,
        hidden_activation=self.activation,
        hidden_init=self.default_init,
        skips=self.skips)

    branches = {
        'rotation':
            MLP(depth=self.rotation_depth,
                width=self.rotation_width,
                hidden_activation=self.activation,
                hidden_init=self.default_init,
                output_init=self.rotation_init,
                output_channels=3)
    }
    if self.use_pivot:
      branches['pivot'] = MLP(
          depth=self.pivot_depth,
          width=self.pivot_width,
          hidden_activation=self.activation,
          hidden_init=self.default_init,
          output_init=self.pivot_init,
          output_channels=3)
    if self.use_translation:
      branches['translation'] = MLP(
          depth=self.translation_depth,
          width=self.translation_width,
          hidden_activation=self.activation,
          hidden_init=self.default_init,
          output_init=self.translation_init,
          output_channels=3)

    # Note that this must be done this way instead of using mutable operations.
    # See https://github.com/google/flax/issues/524.
    self.branches = branches

  def warp(self,
           points: jnp.ndarray,
           metadata: jnp.ndarray,
           alpha: Optional[float] = None,
           metadata_encoded: bool = False):
    points_embed = self.points_encoder(points, alpha=alpha)
    metadata_embed = (metadata
                      if metadata_encoded
                      else self.metadata_encoder(metadata))
    inputs = jnp.concatenate([points_embed, metadata_embed], axis=-1)
    trunk_output = self.trunk(inputs)

    # Evaluate branches.
    branch_outputs = {}
    for branch_key, branch in self.branches.items():
      branch_outputs[branch_key] = branch(trunk_output)

    # Warp points based on outputs.
    warped_points = points
    if self.use_pivot:
      warped_points = warped_points + branch_outputs['pivot']
    # The rotation matrix is predicted as its log quaternion
    # duals. This is convenient for two reasons:
    #   (1) The exp(log(q)) is guaranteed to be an element of SO(3).
    #   (2) exp(0) is the identity.
    q_rot = quaternion.exp(branch_outputs['rotation'])
    warped_points = quaternion.rotate(q_rot, warped_points)
    if self.use_pivot:
      warped_points = warped_points - branch_outputs['pivot']
    if self.use_translation:
      warped_points = warped_points + branch_outputs['translation']

    return warped_points

  def __call__(self,
               points: jnp.ndarray,
               metadata: jnp.ndarray,
               alpha: Optional[float] = None,
               return_jacobian: bool = False,
               metadata_encoded: bool = False):
    """Warp the given points using a warp field.

    Args:
      points: the points to warp.
      metadata: metadata indices if metadata_encoded is False else pre-encoded
        metadata.
      alpha: the alpha value for the positional encoding.
      return_jacobian: if True compute and return the Jacobian of the warp.
      metadata_encoded: if True assumes the metadata is already encoded.

    Returns:
      The warped points and the Jacobian of the warp if `return_jacobian` is
        True.
    """
    warped_points = self.warp(points, metadata, alpha, metadata_encoded)

    if return_jacobian:
      jac_fn = jax.jacfwd(self.warp, argnums=0)
      jac = jac_fn(points, metadata, alpha, metadata_encoded)
      return warped_points, jac

    return warped_points
