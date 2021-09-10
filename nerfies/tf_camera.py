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

"""A basic camera implementation in Tensorflow."""
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.experimental import numpy as tnp


def _norm(x):
  return tnp.sqrt(tnp.sum(x ** 2, axis=-1, keepdims=True))


def _compute_residual_and_jacobian(
    x: tnp.ndarray,
    y: tnp.ndarray,
    xd: tnp.ndarray,
    yd: tnp.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[tnp.ndarray, tnp.ndarray, tnp.ndarray, tnp.ndarray, tnp.ndarray,
           tnp.ndarray]:
  """Auxiliary function of radial_and_tangential_undistort()."""
  # let r(x, y) = x^2 + y^2;
  #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + k3 * r))

  # The perfect projection is:
  # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
  # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
  #
  # Let's define
  #
  # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
  # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
  #
  # We are looking for a solution that satisfies
  # fx(x, y) = fy(x, y) = 0;
  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: tnp.ndarray,
    yd: tnp.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[tnp.ndarray, tnp.ndarray]:
  """Computes undistorted (x, y) from (xd, yd)."""
  # Initialize from the distorted point.
  x = xd
  y = yd

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = tnp.where(
        tnp.abs(denominator) > eps, x_numerator / denominator,
        tnp.zeros_like(denominator))
    step_y = tnp.where(
        tnp.abs(denominator) > eps, y_numerator / denominator,
        tnp.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

  return x, y


class TFCamera:
  """A duplicate of our JAX-basded camera class.

  This is necessary to use tf.data.Dataset.
  """

  def __init__(self,
               orientation: tnp.ndarray,
               position: tnp.ndarray,
               focal_length: float,
               principal_point: tnp.ndarray,
               image_size: tnp.ndarray,
               skew: float = 0.0,
               pixel_aspect_ratio: float = 1.0,
               radial_distortion: Optional[tnp.ndarray] = None,
               tangential_distortion: Optional[tnp.ndarray] = None,
               dtype=tnp.float32):
    """Constructor for camera class."""
    if radial_distortion is None:
      radial_distortion = tnp.array([0.0, 0.0, 0.0], dtype)
    if tangential_distortion is None:
      tangential_distortion = tnp.array([0.0, 0.0], dtype)

    self.orientation = tnp.array(orientation, dtype)
    self.position = tnp.array(position, dtype)
    self.focal_length = tnp.array(focal_length, dtype)
    self.principal_point = tnp.array(principal_point, dtype)
    self.skew = tnp.array(skew, dtype)
    self.pixel_aspect_ratio = tnp.array(pixel_aspect_ratio, dtype)
    self.radial_distortion = tnp.array(radial_distortion, dtype)
    self.tangential_distortion = tnp.array(tangential_distortion, dtype)
    self.image_size = tnp.array(image_size, dtype)
    self.dtype = dtype

  @property
  def scale_factor_x(self):
    return self.focal_length

  @property
  def scale_factor_y(self):
    return self.focal_length * self.pixel_aspect_ratio

  @property
  def principal_point_x(self):
    return self.principal_point[0]

  @property
  def principal_point_y(self):
    return self.principal_point[1]

  @property
  def image_size_y(self):
    return self.image_size[1]

  @property
  def image_size_x(self):
    return self.image_size[0]

  @property
  def image_shape(self):
    return self.image_size_y, self.image_size_x

  @property
  def optical_axis(self):
    return self.orientation[2, :]

  def pixel_to_local_rays(self, pixels: tnp.ndarray):
    """Returns the local ray directions for the provided pixels."""
    y = ((pixels[..., 1] - self.principal_point_y) / self.scale_factor_y)
    x = ((pixels[..., 0] - self.principal_point_x - y * self.skew) /
         self.scale_factor_x)

    x, y = _radial_and_tangential_undistort(
        x,
        y,
        k1=self.radial_distortion[0],
        k2=self.radial_distortion[1],
        k3=self.radial_distortion[2],
        p1=self.tangential_distortion[0],
        p2=self.tangential_distortion[1])

    dirs = tnp.stack([x, y, tnp.ones_like(x)], axis=-1)
    return dirs / _norm(dirs)

  def pixels_to_rays(self,
                     pixels: tnp.ndarray) -> Tuple[tnp.ndarray, tnp.ndarray]:
    """Returns the rays for the provided pixels.

    Args:
      pixels: [A1, ..., An, 2] tensor or np.array containing 2d pixel positions.

    Returns:
        An array containing the normalized ray directions in world coordinates.
    """
    if pixels.shape[-1] != 2:
      raise ValueError('The last dimension of pixels must be 2.')
    if pixels.dtype != self.dtype:
      raise ValueError(f'pixels dtype ({pixels.dtype!r}) must match camera '
                       f'dtype ({self.dtype!r})')

    local_rays_dir = self.pixel_to_local_rays(pixels)
    rays_dir = tf.linalg.matvec(
        self.orientation, local_rays_dir, transpose_a=True)

    # Normalize rays.
    rays_dir = rays_dir / _norm(rays_dir)
    return rays_dir

  def pixels_to_points(self, pixels: tnp.ndarray, depth: tnp.ndarray):
    rays_through_pixels = self.pixels_to_rays(pixels)
    cosa = rays_through_pixels @ self.optical_axis
    points = (
        rays_through_pixels * depth[..., tnp.newaxis] / cosa[..., tnp.newaxis] +
        self.position)
    return points

  def points_to_local_points(self, points: tnp.ndarray):
    translated_points = points - self.position
    local_points = (self.orientation @ translated_points.T).T
    return local_points

  def get_pixel_centers(self):
    """Returns the pixel centers."""
    xx, yy = tf.meshgrid(tf.range(self.image_size_x),
                         tf.range(self.image_size_y))
    return tf.cast(tf.stack([xx, yy], axis=-1), self.dtype) + 0.5
