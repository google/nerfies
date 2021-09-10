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

"""Modules for NeRF models."""
import functools
from typing import Optional, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp

from nerfies import types


class MLP(nn.Module):
  """Basic MLP class with hidden layers and an output layers."""
  depth: int
  width: int
  hidden_init: types.Initializer = nn.initializers.xavier_uniform()
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


class NerfMLP(nn.Module):
  """A simple MLP.

  Attributes:
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_rgb_branch_depth: int, the depth of the second part of MLP.
    nerf_rgb_branch_width: int, the width of the second part of MLP.
    activation: function, the activation function used in the MLP.
    skips: which layers to add skip layers to.
    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    condition_density: if True put the condition at the begining which
      conditions the density of the field.
  """
  trunk_depth: int = 8
  trunk_width: int = 256

  rgb_branch_depth: int = 1
  rgb_branch_width: int = 128
  rgb_channels: int = 3

  alpha_branch_depth: int = 0
  alpha_branch_width: int = 128
  alpha_channels: int = 1

  activation: types.Activation = nn.relu
  skips: Tuple[int] = (4,)

  @nn.compact
  def __call__(self, x, trunk_condition, alpha_condition, rgb_condition):
    """Multi-layer perception for nerf.

    Args:
      x: sample points with shape [batch, num_coarse_samples, feature].
      trunk_condition: a condition array provided to the trunk.
      alpha_condition: a condition array provided to the alpha branch.
      rgb_condition: a condition array provided in the RGB branch.

    Returns:
      raw: [batch, num_coarse_samples, rgb_channels+alpha_channels].
    """
    dense = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())

    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])

    def broadcast_condition(c):
      # Broadcast condition from [batch, feature] to
      # [batch, num_coarse_samples, feature] since all the samples along the
      # same ray has the same viewdir.
      c = jnp.tile(c[:, None, :], (1, num_samples, 1))
      # Collapse the [batch, num_coarse_samples, feature] tensor to
      # [batch * num_coarse_samples, feature] to be fed into nn.Dense.
      c = c.reshape([-1, c.shape[-1]])
      return c

    trunk_mlp = MLP(depth=self.trunk_depth,
                    width=self.trunk_width,
                    hidden_activation=self.activation,
                    hidden_init=jax.nn.initializers.glorot_uniform(),
                    skips=self.skips)
    rgb_mlp = MLP(depth=self.rgb_branch_depth,
                  width=self.rgb_branch_width,
                  hidden_activation=self.activation,
                  hidden_init=jax.nn.initializers.glorot_uniform(),
                  output_init=jax.nn.initializers.glorot_uniform(),
                  output_channels=self.rgb_channels)
    alpha_mlp = MLP(depth=self.alpha_branch_depth,
                    width=self.alpha_branch_width,
                    hidden_activation=self.activation,
                    hidden_init=jax.nn.initializers.glorot_uniform(),
                    output_init=jax.nn.initializers.glorot_uniform(),
                    output_channels=self.alpha_channels)

    if trunk_condition is not None:
      trunk_condition = broadcast_condition(trunk_condition)
      trunk_input = jnp.concatenate([x, trunk_condition], axis=-1)
    else:
      trunk_input = x
    x = trunk_mlp(trunk_input)

    if (alpha_condition is not None) or (rgb_condition is not None):
      bottleneck = dense(self.trunk_width, name='bottleneck')(x)

    if alpha_condition is not None:
      alpha_condition = broadcast_condition(alpha_condition)
      alpha_input = jnp.concatenate([bottleneck, alpha_condition], axis=-1)
    else:
      alpha_input = x
    alpha = alpha_mlp(alpha_input)

    if rgb_condition is not None:
      rgb_condition = broadcast_condition(rgb_condition)
      rgb_input = jnp.concatenate([bottleneck, rgb_condition], axis=-1)
    else:
      rgb_input = x
    rgb = rgb_mlp(rgb_input)

    return {
        'rgb': rgb.reshape((-1, num_samples, self.rgb_channels)),
        'alpha': alpha.reshape((-1, num_samples, self.alpha_channels)),
    }


class SinusoidalEncoder(nn.Module):
  """A vectorized sinusoidal encoding.

  Attributes:
    num_freqs: the number of frequency bands in the encoding.
    min_freq_log2: the log (base 2) of the lower frequency.
    max_freq_log2: the log (base 2) of the upper frequency.
    scale: a scaling factor for the positional encoding.
    use_identity: if True use the identity encoding as well.
  """
  num_freqs: int
  min_freq_log2: int = 0
  max_freq_log2: Optional[int] = None
  scale: float = 1.0
  use_identity: bool = True

  def setup(self):
    if self.max_freq_log2 is None:
      max_freq_log2 = self.num_freqs - 1.0
    else:
      max_freq_log2 = self.max_freq_log2
    self.freq_bands = 2.0**jnp.linspace(self.min_freq_log2,
                                        max_freq_log2,
                                        int(self.num_freqs))

    # (F, 1).
    self.freqs = jnp.reshape(self.freq_bands, (self.num_freqs, 1))

  def __call__(self, x, alpha: Optional[float] = None):
    """A vectorized sinusoidal encoding.

    Args:
      x: the input features to encode.
      alpha: a dummy argument for API compatibility.

    Returns:
      A tensor containing the encoded features.
    """
    if self.num_freqs == 0:
      return x

    x_expanded = jnp.expand_dims(x, axis=-2)  # (1, C).
    # Will be broadcasted to shape (F, C).
    angles = self.scale * x_expanded * self.freqs

    # The shape of the features is (F, 2, C) so that when we reshape it
    # it matches the ordering of the original NeRF code.
    # Vectorize the computation of the high-frequency (sin, cos) terms.
    # We use the trigonometric identity: cos(x) = sin(x + pi/2)
    features = jnp.stack((angles, angles + jnp.pi / 2), axis=-2)
    features = features.flatten()
    features = jnp.sin(features)

    # Prepend the original signal for the identity.
    if self.use_identity:
      features = jnp.concatenate([x, features], axis=-1)
    return features


class AnnealedSinusoidalEncoder(nn.Module):
  """An annealed sinusoidal encoding."""
  num_freqs: int
  min_freq_log2: int = 0
  max_freq_log2: Optional[int] = None
  scale: float = 1.0
  use_identity: bool = True

  @nn.compact
  def __call__(self, x, alpha):
    if alpha is None:
      raise ValueError('alpha must be specified.')
    if self.num_freqs == 0:
      return x

    num_channels = x.shape[-1]

    base_encoder = SinusoidalEncoder(
        num_freqs=self.num_freqs,
        min_freq_log2=self.min_freq_log2,
        max_freq_log2=self.max_freq_log2,
        scale=self.scale,
        use_identity=self.use_identity)
    features = base_encoder(x)

    if self.use_identity:
      identity, features = jnp.split(features, (x.shape[-1],), axis=-1)

    # Apply the window by broadcasting to save on memory.
    features = jnp.reshape(features, (-1, 2, num_channels))
    window = self.cosine_easing_window(
        self.min_freq_log2, self.max_freq_log2, self.num_freqs, alpha)
    window = jnp.reshape(window, (-1, 1, 1))
    features = window * features

    if self.use_identity:
      return jnp.concatenate([
          identity,
          features.flatten(),
      ], axis=-1)
    else:
      return features.flatten()

  @classmethod
  def cosine_easing_window(cls, min_freq_log2, max_freq_log2, num_bands, alpha):
    """Eases in each frequency one by one with a cosine.

    This is equivalent to taking a Tukey window and sliding it to the right
    along the frequency spectrum.

    Args:
      min_freq_log2: the lower frequency band.
      max_freq_log2: the upper frequency band.
      num_bands: the number of frequencies.
      alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

    Returns:
      A 1-d numpy array with num_sample elements containing the window.
    """
    if max_freq_log2 is None:
      max_freq_log2 = num_bands - 1.0
    bands = jnp.linspace(min_freq_log2, max_freq_log2, num_bands)
    x = jnp.clip(alpha - bands, 0.0, 1.0)
    return 0.5 * (1 + jnp.cos(jnp.pi * x + jnp.pi))


class TimeEncoder(nn.Module):
  """Encodes a timestamp to an embedding."""
  num_freqs: int

  features: int = 10
  depth: int = 6
  width: int = 64
  skips: int = (4,)
  hidden_init: types.Initializer = nn.initializers.xavier_uniform()
  output_init: types.Activation = nn.initializers.uniform(scale=0.05)

  def setup(self):
    self.posenc = AnnealedSinusoidalEncoder(num_freqs=self.num_freqs)
    self.mlp = MLP(
        depth=self.depth,
        width=self.width,
        skips=self.skips,
        hidden_init=self.hidden_init,
        output_channels=self.features,
        output_init=self.output_init)

  def __call__(self, time, alpha=None):
    if alpha is None:
      alpha = self.num_freqs
    encoded_time = self.posenc(time, alpha)
    return self.mlp(encoded_time)
