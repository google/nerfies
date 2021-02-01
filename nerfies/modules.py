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


class NerfMLP(nn.Module):
  """A simple MLP.

  Attributes:
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_condition_depth: int, the depth of the second part of MLP.
    nerf_condition_width: int, the width of the second part of MLP.
    activation: function, the activation function used in the MLP.
    skips: which layers to add skip layers to.
    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
  """
  nerf_trunk_depth: int = 8
  nerf_trunk_width: int = 256
  nerf_condition_depth: int = 1
  nerf_condition_width: int = 128
  activation: types.Activation = nn.relu
  skips: Tuple[int] = (4,)
  alpha_channels: int = 1
  rgb_channels: int = 3

  @nn.compact
  def __call__(self, x, condition):
    """Multi-layer perception for nerf.

    Args:
      x: sample points with shape [batch, num_coarse_samples, feature].
      condition: condition array of shape [batch, feature], if not None, this
        variable will be part of the input to the second part of the MLP
        concatenated with the output vector of the first part of the MLP. If
        None, only the first part of the MLP will be used with input x. In the
        original paper, this variable is the view direction.

    Returns:
      raw: [batch, num_coarse_samples, rgb_channels+alpha_channels].
    """
    dense = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())

    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])

    inputs = x
    for i in range(self.nerf_trunk_depth):
      x = dense(self.nerf_trunk_width, name=f'trunk_{i}')(x)
      x = self.activation(x)
      if i in self.skips:
        x = jnp.concatenate([inputs, x], axis=-1)
    alpha = dense(self.alpha_channels, name='alpha_logits')(x)
    if condition is not None:
      # Output of the first part of MLP.
      bottleneck = dense(self.nerf_trunk_width, name='bottleneck')(x)
      # Broadcast condition from [batch, feature] to
      # [batch, num_coarse_samples, feature] since all the samples along the
      # same ray has the same viewdir.
      condition = jnp.tile(condition[:, None, :], (1, num_samples, 1))
      # Collapse the [batch, num_coarse_samples, feature] tensor to
      # [batch * num_coarse_samples, feature] to be fed into nn.Dense.
      condition = condition.reshape([-1, condition.shape[-1]])
      x = jnp.concatenate([bottleneck, condition], axis=-1)
      # Here use 1 extra layer to align with the original nerf model.
      for i in range(self.nerf_condition_depth):
        x = dense(self.nerf_condition_width, name=f'condition_{i}')(x)
        x = self.activation(x)
    rgb = dense(self.rgb_channels, name='rgb_logits')(x)
    return {
        'rgb': rgb.reshape((-1, num_samples, self.rgb_channels)),
        'alpha': alpha.reshape((-1, num_samples, self.alpha_channels)),
    }


class SinusoidalEncoder(nn.Module):
  """A vectorized sinusoidal encoding.

  Attributes:
    num_freqs: the number of frequency bands in the encoding.
    max_freq_log2: the log (base 2) of the maximum frequency.
    scale: a scaling factor for the positional encoding.
    use_identity: if True use the identity encoding as well.
  """
  num_freqs: int
  max_freq_log2: Optional[int] = None
  scale: float = 1.0
  use_identity: bool = True

  def setup(self):
    if self.max_freq_log2 is None:
      max_freq_log2 = self.num_freqs - 1.0
    else:
      max_freq_log2 = self.max_freq_log2
    self.freq_bands = 2.0**jnp.linspace(0.0, max_freq_log2, int(self.num_freqs))

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
  max_freq_log2: Optional[int] = None
  scale: float = 1.0

  @nn.compact
  def __call__(self, x, alpha):
    if alpha is None:
      raise ValueError('alpha must be specified.')
    if self.num_freqs == 0:
      return x

    num_channels = x.shape[-1]

    base_encoder = SinusoidalEncoder(
        num_freqs=self.num_freqs,
        max_freq_log2=self.max_freq_log2,
        scale=self.scale)
    features = base_encoder(x)
    identity, features = jnp.split(features, (x.shape[-1],), axis=-1)

    # Apply the window by broadcasting to save on memory.
    features = jnp.reshape(features, (-1, 2, num_channels))
    window = self.cosine_easing_window(self.num_freqs, alpha)
    window = jnp.reshape(window, (-1, 1, 1))
    features = window * features

    return jnp.concatenate([
        identity,
        features.flatten(),
    ], axis=-1)

  @classmethod
  def cosine_easing_window(cls, num_freqs, alpha):
    """Eases in each frequency one by one with a cosine.

    This is equivalent to taking a Tukey window and sliding it to the right
    along the frequency spectrum.

    Args:
      num_freqs: the number of frequencies.
      alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

    Returns:
      A 1-d numpy array with num_sample elements containing the window.
    """
    x = jnp.clip(alpha - jnp.arange(num_freqs, dtype=jnp.float32), 0.0, 1.0)
    return 0.5 * (1 + jnp.cos(jnp.pi * x + jnp.pi))
