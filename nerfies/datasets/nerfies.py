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

"""Casual Volumetric Capture datasets."""
import json
from typing import List, Tuple

from absl import logging
import cv2
import numpy as np

from nerfies import gpath
from nerfies import types
from nerfies import utils
from nerfies.datasets import core


def load_scene_info(
    data_dir: types.PathType) -> Tuple[np.ndarray, float, float, float]:
  """Loads the scene scale from scene_scale.npy.

  Args:
    data_dir: the path to the dataset.

  Returns:
    scene_center: the center of the scene (unscaled coordinates).
    scene_scale: the scale of the scene.
    near: the near plane of the scene (scaled coordinates).
    far: the far plane of the scene (scaled coordinates).

  Raises:
    ValueError if scene_scale.npy does not exist.
  """
  scene_json_path = gpath.GPath(data_dir, 'scene.json')
  with scene_json_path.open('r') as f:
    scene_json = json.load(f)

  scene_center = np.array(scene_json['center'])
  scene_scale = scene_json['scale']
  near = scene_json['near']
  far = scene_json['far']

  return scene_center, scene_scale, near, far


def _load_image(path: types.PathType) -> np.ndarray:
  path = gpath.GPath(path)
  with path.open('rb') as f:
    raw_im = np.asarray(bytearray(f.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_im, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
    image = np.asarray(image).astype(np.float32) / 255.0
    return image


def _load_dataset_ids(data_dir: types.PathType) -> Tuple[List[str], List[str]]:
  """Loads dataset IDs."""
  dataset_json_path = gpath.GPath(data_dir, 'dataset.json')
  logging.info('*** Loading dataset IDs from %s', dataset_json_path)
  with dataset_json_path.open('r') as f:
    dataset_json = json.load(f)
    train_ids = dataset_json['train_ids']
    val_ids = dataset_json['val_ids']

  train_ids = [str(i) for i in train_ids]
  val_ids = [str(i) for i in val_ids]

  return train_ids, val_ids


class NerfiesDataSource(core.DataSource):
  """Data loader for videos."""

  def __init__(
      self,
      data_dir,
      image_scale: int,
      shuffle_pixels=False,
      camera_type='json',
      test_camera_trajectory='orbit-extreme',
      **kwargs):
    self.data_dir = gpath.GPath(data_dir)
    # Load IDs from JSON if it exists. This is useful since COLMAP fails on
    # some images so this gives us the ability to skip invalid images.
    train_ids, val_ids = _load_dataset_ids(self.data_dir)
    super().__init__(train_ids=train_ids, val_ids=val_ids,
                     **kwargs)
    self.scene_center, self.scene_scale, self._near, self._far = \
      load_scene_info(self.data_dir)
    self.test_camera_trajectory = test_camera_trajectory

    self.image_scale = image_scale
    self.shuffle_pixels = shuffle_pixels

    self.rgb_dir = gpath.GPath(data_dir, 'rgb', f'{image_scale}x')
    self.depth_dir = gpath.GPath(data_dir, 'depth', f'{image_scale}x')
    self.camera_type = camera_type
    self.camera_dir = gpath.GPath(data_dir, 'camera')

    metadata_path = self.data_dir / 'metadata.json'
    self.metadata_dict = None
    if metadata_path.exists():
      with metadata_path.open('r') as f:
        self.metadata_dict = json.load(f)

  @property
  def near(self):
    return self._near

  @property
  def far(self):
    return self._far

  @property
  def camera_ext(self):
    if self.camera_type == 'json':
      return '.json'

    raise ValueError(f'Unknown camera_type {self.camera_type}')

  def get_rgb_path(self, item_id):
    return self.rgb_dir / f'{item_id}.png'

  def load_rgb(self, item_id):
    return _load_image(self.rgb_dir / f'{item_id}.png')

  def load_camera(self, item_id, scale_factor=1.0):
    if isinstance(item_id, gpath.GPath):
      camera_path = item_id
    else:
      if self.camera_type == 'json':
        camera_path = self.camera_dir / f'{item_id}{self.camera_ext}'
      else:
        raise ValueError(f'Unknown camera type {self.camera_type!r}.')

    return core.load_camera(camera_path,
                            scale_factor=scale_factor / self.image_scale,
                            scene_center=self.scene_center,
                            scene_scale=self.scene_scale)

  def glob_cameras(self, path):
    path = gpath.GPath(path)
    return sorted(path.glob(f'*{self.camera_ext}'))

  def load_test_cameras(self, count=None):
    camera_dir = (self.data_dir / 'camera-paths' / self.test_camera_trajectory)
    if not camera_dir.exists():
      logging.warning('test camera path does not exist: %s', str(camera_dir))
      return []
    camera_paths = sorted(camera_dir.glob(f'*{self.camera_ext}'))
    if count is not None:
      stride = max(1, len(camera_paths) // count)
      camera_paths = camera_paths[::stride]
    cameras = utils.parallel_map(self.load_camera, camera_paths)
    return cameras

  def load_points(self, shuffle=False):
    with (self.data_dir / 'points.npy').open('rb') as f:
      points = np.load(f)
    points = (points - self.scene_center) * self.scene_scale
    points = points.astype(np.float32)
    if shuffle:
      logging.info('Shuffling points.')
      shuffled_inds = self.rng.permutation(len(points))
      points = points[shuffled_inds]
    logging.info('Loaded %d points.', len(points))
    return points

  def get_appearance_id(self, item_id):
    return self.metadata_dict[item_id]['appearance_id']

  def get_camera_id(self, item_id):
    return self.metadata_dict[item_id]['camera_id']

  def get_warp_id(self, item_id):
    return self.metadata_dict[item_id]['warp_id']

  def get_time_id(self, item_id):
    if 'time_id' in self.metadata_dict[item_id]:
      return self.metadata_dict[item_id]['time_id']
    else:
      # Fallback for older datasets.
      return self.metadata_dict[item_id]['warp_id']
