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

"""Core dataset methods and classes."""
import abc
import functools
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Iterable

from absl import logging
from flax import jax_utils
import jax
from jax import tree_util
import numpy as np
import tensorflow as tf

from nerfies import camera as cam
from nerfies import gpath
from nerfies import image_utils
from nerfies import utils


def load_camera(camera_path,
                scale_factor=1.0,
                scene_center=None,
                scene_scale=None):
  """Loads camera and rays defined by the center pixels of a camera.

  Args:
    camera_path: a path to a camera.
    scale_factor: a factor to scale the camera image by.
    scene_center: the center of the scene where the camera will be centered to.
    scene_scale: the scale of the scene by which the camera will also be scaled
      by.

  Returns:
    A sfm_camera.Camera instance.
  """
  if camera_path.suffix == '.json':
    camera = cam.Camera.from_json(camera_path)
  else:
    raise ValueError('File must have extension .pb or .json.')

  if scale_factor != 1.0:
    camera = camera.scale(scale_factor)

  if scene_center is not None:
    camera.position = camera.position - scene_center
  if scene_scale is not None:
    camera.position = camera.position * scene_scale

  return camera


def camera_to_rays(
    camera: cam.Camera,
    image_shape: Optional[Tuple[int, int]] = None
) -> Dict[str, np.ndarray]:
  """Converts a vision sfm camera into rays.

  Args:
    camera: the camera to convert to rays.
    image_shape: force the shape of the image. Default None.

  Returns:
    A dictionary of rays. Contains:
      `origins`: the origin of each ray.
      `directions`: unit vectors representing the direction of each ray.
      `pixels`: the pixel centers of each ray.
  """
  camera = camera.copy()

  if not image_shape:
    image_shape = camera.image_shape

  img_rays_origin = np.tile(camera.position[None, None, :],
                            image_shape + (1,))
  img_rays_dir = camera.pixels_to_rays(camera.get_pixel_centers())
  img_rays_pixels = camera.get_pixel_centers()

  return {
      'origins': img_rays_origin.astype(np.float32),
      'directions': img_rays_dir.astype(np.float32),
      'pixels': img_rays_pixels.astype(np.float32),
  }


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def prepare_tf_data_unbatched(xs):
  """Prepare TF dataset into unbatched numpy arrays."""
  # Use _numpy() for zero-copy conversion between TF and NumPy.
  # pylint: disable=protected-access
  return jax.tree_map(lambda x: x._numpy(), xs)


def dataset_from_dict(data_dict, rng, shuffle=False, flatten=False):
  """Crates a dataset from a rays dictionary."""
  num_examples = data_dict['origins'].shape[0]
  heights = [x.shape[0] for x in data_dict['origins']]
  widths = [x.shape[1] for x in data_dict['origins']]

  # Broadcast appearance ID to match ray shapes.
  if 'metadata' in data_dict:
    for metadata_key, metadata in data_dict['metadata'].items():
      data_dict['metadata'][metadata_key] = np.asarray([
          np.full((heights[i], widths[i], 1), fill_value=x)
          for i, x in enumerate(metadata)
      ])

  num_rays = int(sum([x * y for x, y in zip(heights, widths)]))
  shuffled_inds = rng.permutation(num_rays)

  logging.info(
      '*** Loaded dataset items: num_rays=%d, num_examples=%d',
      num_rays, num_examples)

  def _prepare_array(x):
    if not isinstance(x, np.ndarray):
      x = np.asarray(x)
    # Create last dimension if it doesn't exist.
    # The `and` part of the check ensures we're not touching ragged arrays.
    if x.ndim == 1 and x[0].ndim == 0:
      x = np.expand_dims(x, -1)
    if flatten:
      x = np.concatenate([x.reshape(-1, x.shape[-1]) for x in x], axis=0)
    if shuffle:
      x = x[shuffled_inds]
    return x

  out_dict = {}
  for key, value in data_dict.items():
    out_dict[key] = tree_util.tree_map(_prepare_array, value)

  return tf.data.Dataset.from_tensor_slices(out_dict)


def iterator_from_dataset(dataset: tf.data.Dataset,
                          batch_size: int,
                          repeat: bool = True,
                          prefetch_size: int = 0,
                          devices: Optional[Sequence[Any]] = None):
  """Create a data iterator that returns JAX arrays from a TF dataset.

  Args:
    dataset: the dataset to iterate over.
    batch_size: the batch sizes the iterator should return.
    repeat: whether the iterator should repeat the dataset.
    prefetch_size: the number of batches to prefetch to device.
    devices: the devices to prefetch to.

  Returns:
    An iterator that returns data batches.
  """
  if repeat:
    dataset = dataset.repeat()

  if batch_size > 0:
    dataset = dataset.batch(batch_size)
    it = map(prepare_tf_data, dataset)
  else:
    it = map(prepare_tf_data_unbatched, dataset)

  if prefetch_size > 0:
    it = jax_utils.prefetch_to_device(it, prefetch_size, devices)

  return it


class DataSource(abc.ABC):
  """An abstract class that defines a data source."""

  def __init__(self,
               use_appearance_id=False,
               use_camera_id=False,
               use_warp_id=False,
               use_depth=False,
               use_relative_depth=False,
               random_seed=0,
               **_):
    self.use_appearance_id = use_appearance_id
    self.use_camera_id = use_camera_id
    self.use_warp_id = use_warp_id
    self.use_depth = use_depth
    self.use_relative_depth = use_relative_depth
    self.rng = np.random.RandomState(random_seed)

  @property
  @abc.abstractmethod
  def train_ids(self):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def val_ids(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def load_rgb(self, item_id):
    raise NotImplementedError()

  def load_depth(self, item_id):
    raise NotImplementedError()

  def load_relative_depth(self, item_id):
    raise NotImplementedError()

  @abc.abstractmethod
  def load_camera(self, item_id, scale_factor=1.0):
    raise NotImplementedError()

  def load_points(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_appearance_id(self, item_id):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_camera_id(self, item_id):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_warp_id(self, item_id):
    raise NotImplementedError()

  @property
  def appearance_ids(self):
    if not self.use_appearance_id:
      return []
    return sorted(set([self.get_appearance_id(i) for i in self.train_ids]))

  @property
  def camera_ids(self):
    if not self.use_camera_id:
      return []
    return sorted(set([self.get_camera_id(i) for i in self.train_ids]))

  @property
  def warp_ids(self):
    if not self.use_warp_id:
      return []
    return sorted(set([self.get_warp_id(i) for i in self.train_ids]))

  @property
  def near(self) -> float:
    raise NotImplementedError()

  @property
  def far(self) -> float:
    raise NotImplementedError()

  @abc.abstractmethod
  def load_test_cameras(self, count=None):
    raise NotImplementedError()

  def parallel_get_items(self, item_ids, scale_factor=1.0):
    """Load data dictionaries indexed by indices in parallel."""
    load_fn = functools.partial(self.get_item, scale_factor=scale_factor)
    data_list = utils.parallel_map(load_fn, item_ids)
    data_dict = utils.tree_collate(data_list)
    return data_dict

  def create_cameras_dataset(
      self,
      cameras: Union[Iterable[cam.Camera], Iterable[gpath.GPath]],
      flatten=False,
      shuffle=False):
    if isinstance(cameras[0], gpath.GPath) or isinstance(cameras[0], str):
      cameras = utils.parallel_map(self.load_camera, cameras)
    data_dict = utils.tree_collate([camera_to_rays(c) for c in cameras])
    return dataset_from_dict(data_dict,
                             rng=self.rng,
                             flatten=flatten,
                             shuffle=shuffle)

  def create_dataset(
      self, item_ids, flatten=False, shuffle=False) -> tf.data.Dataset:
    """Create a Tensorflow Dataset from a data dictionary."""
    logging.info('*** Creating a dataset with %d items.', len(item_ids))
    data_dict = self.parallel_get_items(item_ids)
    return dataset_from_dict(data_dict,
                             rng=self.rng,
                             flatten=flatten,
                             shuffle=shuffle)

  def create_iterator(self,
                      item_ids,
                      batch_size: int,
                      repeat: bool = True,
                      flatten=False,
                      shuffle=False,
                      prefetch_size: int = 0,
                      devices: Optional[Sequence[Any]] = None):
    dataset = self.create_dataset(item_ids, flatten=flatten, shuffle=shuffle)
    return iterator_from_dataset(dataset=dataset,
                                 batch_size=batch_size,
                                 repeat=repeat,
                                 prefetch_size=prefetch_size,
                                 devices=devices)

  def get_item(self, item_id, scale_factor=1.0):
    """Load an example as a data dictionary.

    Args:
      item_id: the ID of the item to fetch.
      scale_factor: a scale factor to apply to the camera.

    Returns:
      A dictionary containing one of more of the following items:
        `rgb`: the RGB pixel values of each ray.
        `rays_dir`: the direction of each ray.
        `rays_origin`: the origin of each ray.
        `rays_pixels`: the pixel center of each ray.
        `metadata`: a dictionary of containing various metadata arrays. Each
          item is an array containing metadata IDs for each ray.
    """
    rgb = self.load_rgb(item_id)
    if scale_factor != 1.0:
      rgb = image_utils.rescale_image(rgb, scale_factor)

    camera = self.load_camera(item_id, scale_factor)
    rays_dict: Dict[str, Any] = camera_to_rays(camera)
    rays_dict['rgb'] = rgb
    rays_dict['metadata'] = {}

    if self.use_appearance_id:
      rays_dict['metadata']['appearance'] = self.get_appearance_id(item_id)
    if self.use_camera_id:
      rays_dict['metadata']['camera'] = self.get_camera_id(item_id)
    if self.use_warp_id:
      rays_dict['metadata']['warp'] = self.get_warp_id(item_id)

    if self.use_depth:
      depth = self.load_depth(item_id)
      if depth is not None:
        if scale_factor != 1.0:
          depth = image_utils.rescale_image(depth, scale_factor)
        rays_dict['depth'] = depth[..., np.newaxis]

    if self.use_relative_depth:
      rel_depth = self.load_relative_depth(item_id)
      if rel_depth is not None:
        if scale_factor != 1.0:
          rel_depth = image_utils.rescale_image(rel_depth, scale_factor)
        rays_dict['rel_depth'] = rel_depth[..., np.newaxis]

    logging.info(
        '\tLoaded item %s: shape=%s, scale_factor=%f, metadata=%s',
        item_id,
        rgb.shape,
        scale_factor,
        str(rays_dict.get('metadata')))

    return rays_dict
