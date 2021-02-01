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

"""Class for handling cameras."""
import copy
import json
from typing import Tuple

from jax import numpy as jnp

from nerfies import gpath
from nerfies import types


def _compute_residual_and_jacobian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    xd: jnp.ndarray,
    yd: jnp.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray]:
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
    xd: jnp.ndarray,
    yd: jnp.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes undistorted (x, y) from (xd, yd)."""
  # Initialize from the distorted point.
  x = xd.copy()
  y = yd.copy()

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = jnp.where(
        jnp.abs(denominator) > eps, x_numerator / denominator,
        jnp.zeros_like(denominator))
    step_y = jnp.where(
        jnp.abs(denominator) > eps, y_numerator / denominator,
        jnp.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

  return x, y


class Camera:
  """Class to handle camera geometry."""

  def __init__(self,
               orientation: jnp.ndarray,
               position: jnp.ndarray,
               focal_length: float,
               principal_point: jnp.ndarray,
               skew: float,
               pixel_aspect_ratio: float,
               radial_distortion: jnp.ndarray,
               tangential_distortion: jnp.ndarray,
               image_size: jnp.ndarray,
               dtype=jnp.float32):
    """Constructor for camera class."""
    self.orientation = orientation
    self.position = position
    self.focal_length = focal_length
    self.principal_point = principal_point
    self.skew = skew
    self.pixel_aspect_ratio = pixel_aspect_ratio
    self.radial_distortion = radial_distortion
    self.tangential_distortion = tangential_distortion
    self.image_size = image_size
    self.dtype = dtype

  @classmethod
  def from_json(cls, path: types.PathType):
    """Loads a JSON camera into memory."""
    path = gpath.GPath(path)
    with path.open('r') as fp:
      camera_json = json.load(fp)

    return cls(
        orientation=jnp.asarray(camera_json['orientation']),
        position=jnp.asarray(camera_json['position']),
        focal_length=camera_json['focal_length'],
        principal_point=jnp.asarray(camera_json['principal_point']),
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=jnp.asarray(camera_json['radial_distortion']),
        tangential_distortion=jnp.asarray(camera_json['tangential']),
        image_size=jnp.asarray(camera_json['image_size']),
    )

  def to_json(self):
    return {
        'orientation': self.orientation.tolist(),
        'position': self.position.tolist(),
        'focal_length': self.focal_length,
        'principal_point': self.principal_point.tolist(),
        'skew': self.skew,
        'pixel_aspect_ratio': self.pixel_aspect_ratio,
        'radial_distortion': self.radial_distortion.tolist(),
        'tangential': self.tangential_distortion.tolist(),
        'image_size': self.image_size.tolist(),
    }

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
  def has_tangential_distortion(self):
    return any(self.tangential_distortion != 0.0)

  @property
  def has_radial_distortion(self):
    return any(self.radial_distortion != 0.0)

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

  def pixel_to_local_rays(self, pixels: jnp.ndarray):
    """Returns the local ray directions for the provided pixels."""
    y = ((pixels[..., 1] - self.principal_point_y) / self.scale_factor_y)
    x = ((pixels[..., 0] - self.principal_point_x - y * self.skew) /
         self.scale_factor_x)

    if self.has_radial_distortion or self.has_tangential_distortion:
      x, y = _radial_and_tangential_undistort(
          x,
          y,
          k1=self.radial_distortion[0],
          k2=self.radial_distortion[1],
          k3=self.radial_distortion[2],
          p1=self.tangential_distortion[0],
          p2=self.tangential_distortion[1])

    dirs = jnp.stack([x, y, jnp.ones_like(x)], axis=-1)
    return dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

  def pixels_to_rays(self,
                     pixels: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    batch_shape = pixels.shape[:-1]
    pixels = pixels.reshape((-1, 2))

    local_rays_dir = self.pixel_to_local_rays(pixels)
    rays_dir = self.orientation.T @ local_rays_dir[..., jnp.newaxis]
    rays_dir = jnp.squeeze(rays_dir, axis=-1)

    # Normalize rays.
    rays_dir /= jnp.linalg.norm(rays_dir, axis=-1, keepdims=True)
    rays_dir = rays_dir.reshape((*batch_shape, 3))
    return rays_dir

  def pixels_to_points(self, pixels: jnp.ndarray, depth: jnp.ndarray):
    rays_through_pixels = self.pixels_to_rays(pixels)
    cosa = rays_through_pixels @ self.optical_axis
    points = (
        rays_through_pixels * depth[..., jnp.newaxis] / cosa[..., jnp.newaxis] +
        self.position)
    return points

  def points_to_local_points(self, points: jnp.ndarray):
    translated_points = points - self.position
    local_points = (self.orientation @ translated_points.T).T
    return local_points

  def project(self, points: jnp.ndarray):
    """Projects a 3D point (x,y,z) to a pixel position (x,y)."""
    batch_shape = points.shape[:-1]
    points = points.reshape((-1, 3))
    local_points = self.points_to_local_points(points)

    # Get normalized local pixel positions.
    x = local_points[..., 0] / local_points[..., 2]
    y = local_points[..., 1] / local_points[..., 2]
    r2 = x**2 + y**2

    # Apply radial distortion.
    distortion = 1.0 + r2 * (
        self.radial_distortion[0] + r2 *
        (self.radial_distortion[1] + self.radial_distortion[2] * r2))

    # Apply tangential distortion.
    x_times_y = x * y
    x = (
        x * distortion + 2.0 * self.tangential_distortion[0] * x_times_y +
        self.tangential_distortion[1] * (r2 + 2.0 * x**2))
    y = (
        y * distortion + 2.0 * self.tangential_distortion[1] * x_times_y +
        self.tangential_distortion[0] * (r2 + 2.0 * y**2))

    # Map the distorted ray to the image plane and return the depth.
    pixel_x = self.focal_length * x + self.skew * y + self.principal_point_x
    pixel_y = self.focal_length * self.pixel_aspect_ratio * y + self.principal_point_y

    pixels = jnp.stack([pixel_x, pixel_y], axis=-1)
    return pixels.reshape((*batch_shape, 2))

  def get_pixel_centers(self):
    """Returns the pixel centers."""
    shape = self.image_shape
    return jnp.moveaxis(jnp.indices(shape, dtype=self.dtype)[::-1], 0, -1) + 0.5

  def scale(self, scale: float):
    """Scales the camera."""
    if scale <= 0:
      raise ValueError('scale needs to be positive.')

    new_camera = Camera(
        orientation=self.orientation.copy(),
        position=self.position.copy(),
        focal_length=self.focal_length * scale,
        principal_point=self.principal_point.copy() * scale,
        skew=self.skew,
        pixel_aspect_ratio=self.pixel_aspect_ratio,
        radial_distortion=self.radial_distortion.copy(),
        tangential_distortion=self.tangential_distortion.copy(),
        image_size=jnp.array((int(round(self.image_size[0] * scale)),
                              int(round(self.image_size[1] * scale)))),
    )
    return new_camera

  def copy(self):
    return copy.deepcopy(self)
