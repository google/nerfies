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

"""Module for evaluating a trained NeRF."""
import time

from absl import logging
import jax
from jax import tree_util
import jax.numpy as jnp

from nerfies import utils


def render_image(state, rays_dict, model_fn, device_count, rng, chunk=8192):
  """Render all the pixels of an image (in test mode).

  Args:
    state: model_utils.TrainState.
    rays_dict: dict, test example.
    model_fn: function, jit-ed render function.
    device_count: The number of devices to shard batches over.
    rng: The random number generator.
    chunk: int, the size of chunks to render sequentially.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    depth: jnp.ndarray, rendered depth.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  h, w = rays_dict['origins'].shape[:2]
  rays_dict = tree_util.tree_map(lambda x: x.reshape((h * w, -1)), rays_dict)
  num_rays = h * w
  _, key_0, key_1 = jax.random.split(rng, 3)
  key_0 = jax.random.split(key_0, device_count)
  key_1 = jax.random.split(key_1, device_count)
  host_id = jax.host_id()
  rgb = []
  depth_exp = []
  depth_med = []
  acc = []
  start_time = time.time()
  for i in range(0, num_rays, chunk):
    logging.info('\tRendering ray batch: %d/%d', i, num_rays)
    # pylint: disable=cell-var-from-loop
    chunk_slice_fn = lambda x: x[i:i + chunk]
    chunk_rays_dict = tree_util.tree_map(chunk_slice_fn, rays_dict)
    num_chunk_rays = chunk_rays_dict['origins'].shape[0]
    remainder = num_chunk_rays % device_count
    if remainder != 0:
      padding = device_count - remainder
      # pylint: disable=cell-var-from-loop
      chunk_pad_fn = lambda x: jnp.pad(x, ((0, padding), (0, 0)), mode='edge')
      chunk_rays_dict = tree_util.tree_map(chunk_pad_fn, chunk_rays_dict)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # host_count.
    per_host_rays = num_chunk_rays // jax.host_count()
    chunk_rays_dict = tree_util.tree_map(
        lambda x: x[(host_id * per_host_rays):((host_id + 1) * per_host_rays)],
        chunk_rays_dict)
    chunk_rays_dict = utils.shard(chunk_rays_dict, device_count)
    model_out = model_fn(
        key_0,
        key_1,
        state.optimizer.target['model'],
        chunk_rays_dict,
        state.warp_alpha)
    ret_key = 'fine' if 'fine' in model_out else 'coarse'
    rgb.append(utils.unshard(model_out[ret_key]['rgb'][0], padding))
    depth_exp.append(utils.unshard(model_out[ret_key]['depth'][0], padding))
    depth_med.append(utils.unshard(model_out[ret_key]['med_depth'][0], padding))
    acc.append(utils.unshard(model_out[ret_key]['acc'][0], padding))
  rgb = jnp.concatenate(rgb, axis=0)
  depth_exp = jnp.concatenate(depth_exp, axis=0)
  depth_med = jnp.concatenate(depth_med, axis=0)
  acc = jnp.concatenate(acc, axis=0)
  logging.info('Rendering took %.04s', time.time() - start_time)
  return (
      rgb.reshape((h, w, -1)),
      depth_exp.reshape((h, w, -1)),
      depth_med.reshape((h, w, -1)),
      acc.reshape((h, w, -1))
  )
