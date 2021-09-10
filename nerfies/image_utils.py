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

"""Image-related utility functions."""
import math
from typing import Tuple

from absl import logging
import cv2
import imageio
import numpy as np
from PIL import Image

from nerfies import gpath
from nerfies import types


UINT8_MAX = 255
UINT16_MAX = 65535


def make_divisible(image: np.ndarray, divisor: int) -> np.ndarray:
  """Trim the image if not divisible by the divisor."""
  height, width = image.shape[:2]
  if height % divisor == 0 and width % divisor == 0:
    return image

  new_height = height - height % divisor
  new_width = width - width % divisor

  return image[:new_height, :new_width]


def downsample_image(image: np.ndarray, scale: int) -> np.ndarray:
  """Downsamples the image by an integer factor to prevent artifacts."""
  if scale == 1:
    return image

  height, width = image.shape[:2]
  if height % scale > 0 or width % scale > 0:
    raise ValueError(f'Image shape ({height},{width}) must be divisible by the'
                     f' scale ({scale}).')
  out_height, out_width = height // scale, width // scale
  resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
  return resized


def upsample_image(image: np.ndarray, scale: int) -> np.ndarray:
  """Upsamples the image by an integer factor."""
  if scale == 1:
    return image

  height, width = image.shape[:2]
  out_height, out_width = height * scale, width * scale
  resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
  return resized


def reshape_image(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
  """Reshapes the image to the given shape."""
  out_height, out_width = shape
  return cv2.resize(
      image, (out_width, out_height), interpolation=cv2.INTER_AREA)


def rescale_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
  """Resize an image by a scale factor, using integer resizing if possible."""
  scale_factor = float(scale_factor)
  if scale_factor <= 0.0:
    raise ValueError('scale_factor must be a non-negative number.')

  if scale_factor == 1.0:
    return image

  height, width = image.shape[:2]
  if scale_factor.is_integer():
    return upsample_image(image, int(scale_factor))

  inv_scale = 1.0 / scale_factor
  if (inv_scale.is_integer() and (scale_factor * height).is_integer() and
      (scale_factor * width).is_integer()):
    return downsample_image(image, int(inv_scale))

  logging.warning(
      'resizing image by non-integer factor %f, this may lead to artifacts.',
      scale_factor)

  height, width = image.shape[:2]
  out_height = math.ceil(height * scale_factor)
  out_height -= out_height % 2
  out_width = math.ceil(width * scale_factor)
  out_width -= out_width % 2

  return reshape_image(image, (out_height, out_width))


def variance_of_laplacian(image: np.ndarray) -> np.ndarray:
  """Compute the variance of the Laplacian which measure the focus."""
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  return cv2.Laplacian(gray, cv2.CVX_64F).var()


def image_to_uint8(image: np.ndarray) -> np.ndarray:
  """Convert the image to a uint8 array."""
  if image.dtype == np.uint8:
    return image
  if not issubclass(image.dtype.type, np.floating):
    raise ValueError(
        f'Input image should be a floating type but is of type {image.dtype!r}')
  return (image * UINT8_MAX).clip(0.0, UINT8_MAX).astype(np.uint8)


def image_to_uint16(image: np.ndarray) -> np.ndarray:
  """Convert the image to a uint16 array."""
  if image.dtype == np.uint16:
    return image
  if not issubclass(image.dtype.type, np.floating):
    raise ValueError(
        f'Input image should be a floating type but is of type {image.dtype!r}')
  return (image * UINT16_MAX).clip(0.0, UINT16_MAX).astype(np.uint16)


def image_to_float32(image: np.ndarray) -> np.ndarray:
  """Convert the image to a float32 array and scale values appropriately."""
  if image.dtype == np.float32:
    return image

  dtype = image.dtype
  image = image.astype(np.float32)
  if dtype == np.uint8:
    return image / UINT8_MAX
  elif dtype == np.uint16:
    return image / UINT16_MAX
  elif dtype == np.float64:
    return image
  elif dtype == np.float16:
    return image

  raise ValueError(f'Not sure how to handle dtype {dtype}')


def load_image(path: types.PathType) -> np.ndarray:
  """Reads an image."""
  if not isinstance(path, gpath.GPath):
    path = gpath.GPath(path)

  with path.open('rb') as f:
    return imageio.imread(f)


def save_image(path: types.PathType, image: np.ndarray) -> None:
  """Saves the image to disk or gfile."""
  if not isinstance(path, gpath.GPath):
    path = gpath.GPath(path)

  with path.open('wb') as f:
    image = Image.fromarray(np.asarray(image))
    image.save(f, format=path.suffix.lstrip('.'))


def save_depth(path: types.PathType, depth: np.ndarray) -> None:
  save_image(path, image_to_uint16(depth / 1000.0))


def load_depth(path: types.PathType) -> np.ndarray:
  depth = load_image(path)
  if depth.dtype != np.uint16:
    raise ValueError('Depth image must be of type uint16.')
  return image_to_float32(depth) * 1000.0


def checkerboard(h, w, size=8):
  """Creates a checkerboard pattern with height h and width w."""
  i = int(math.ceil(h / (size * 2)))
  j = int(math.ceil(w / (size * 2)))
  pattern = np.kron([[1, 0] * j, [0, 1] * j] * i,
                    np.ones((size, size)))[:h, :w]
  return np.clip(pattern + 0.8, 0.0, 1.0)
