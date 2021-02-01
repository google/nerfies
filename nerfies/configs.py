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

"""Configuration classes."""
import os
from typing import Any, Optional, Tuple, Mapping

import dataclasses
from flax import nn
import frozendict
import gin

from nerfies import types

ScheduleDef = Any

this_path = os.path.dirname(os.path.realpath(__file__))
configs_path = os.path.join(this_path, '../configs')
gin.add_config_file_search_path(configs_path)

gin.config.external_configurable(nn.elu, module='flax.nn')
gin.config.external_configurable(nn.relu, module='flax.nn')
gin.config.external_configurable(nn.leaky_relu, module='flax.nn')
gin.config.external_configurable(nn.tanh, module='flax.nn')
gin.config.external_configurable(nn.sigmoid, module='flax.nn')
gin.config.external_configurable(nn.softplus, module='flax.nn')


@gin.configurable()
@dataclasses.dataclass
class ModelConfig:
  """Parameters for the model."""
  # Sample linearly in disparity rather than depth.
  use_linear_disparity: bool = False
  # Use white as the default background.
  use_white_background: bool = False
  # Use stratified sampling.
  use_stratified_sampling: bool = True
  # Use the sample at infinity.
  use_sample_at_infinity: bool = True
  # The standard deviation of the alpha noise.
  noise_std: Optional[float] = None

  # The depth of the NeRF.
  nerf_trunk_depth: int = 8
  # The width of the NeRF.
  nerf_trunk_width: int = 256
  # The depth of the conditional part of the MLP.
  nerf_condition_depth: int = 1
  # The width of the conditional part of the MLP.
  nerf_condition_width: int = 128
  # The intermediate activation for the NeRF.
  activation: types.Activation = nn.relu
  # The sigma activation for the NeRF.
  sigma_activation: types.Activation = nn.relu
  # Adds a skip connection every N layers.
  nerf_skips: Tuple[int] = (4,)
  # The number of alpha channels.
  alpha_channels: int = 1
  # The number of RGB channels.
  rgb_channels: int = 3
  # The number of positional encodings for points.
  num_nerf_point_freqs: int = 10
  # The number of positional encodings for viewdirs.
  num_nerf_viewdir_freqs: int = 4
  # The number of coarse samples along each ray.
  num_coarse_samples: int = 64
  # The number of fine samples along each ray.
  num_fine_samples: int = 128
  # Whether to use view directions.
  use_viewdirs: bool = True

  # Whether to use the appearance metadata for the conditional branch.
  use_appearance_metadata: bool = False
  # The number of dimensions for the appearance metadata.
  appearance_metadata_dims: int = 8
  # Whether to use the camera metadata for the conditional branch.
  use_camera_metadata: bool = False
  # The number of dimensions for the camera metadata.
  camera_metadata_dims: int = 2

  # Whether to use the warp field.
  use_warp: bool = False
  # The number of frequencies for the warp field.
  num_warp_freqs: int = 8
  # The number of dimensions for the warp metadata.
  num_warp_features: int = 8
  # The type of warp field to use. One of: 'translation', or 'se3'.
  warp_field_type: str = 'translation'
  # Additional keyword arguments to pass to the warp field.
  warp_kwargs: Mapping[str, Any] = frozendict.frozendict()


@gin.configurable()
@dataclasses.dataclass
class ExperimentConfig:
  """Experiment configuration."""
  # A subname for the experiment e.g., for parameter sweeps. If this is set
  # experiment artifacts will be saves to a subdirectory with this name.
  subname: Optional[str] = None
  # The image scale to use for the dataset. Should be a power of 2.
  image_scale: int = 4
  # The random seed used to initialize the RNGs for the experiment.
  random_seed: int = 12345
  # The type of datasource. Either 'nerfies' or 'dynamic_scene'.
  datasource_type: str = 'nerfies'
  # Data source specification.
  datasource_spec: Optional[Mapping[str, Any]] = None


@gin.configurable()
@dataclasses.dataclass
class TrainConfig:
  """Parameters for training."""
  batch_size: int = gin.REQUIRED

  # The definition for the learning rate schedule.
  lr_schedule: ScheduleDef = frozendict.frozendict({
      'type': 'exponential',
      'initial_value': 0.001,
      'final_value': 0.0001,
      'num_steps': 1000000,
  })
  # The maximum number of training steps.
  max_steps: int = 1000000

  # The start value of the warp alpha.
  warp_alpha_schedule: ScheduleDef = frozendict.frozendict({
      'type': 'linear',
      'initial_value': 0.0,
      'final_value': 8.0,
      'num_steps': 80000,
  })

  # Whether to use the elastic regularization loss.
  use_elastic_loss: bool = False
  # The weight of the elastic regularization loss.
  elastic_loss_weight_schedule: ScheduleDef = ('constant', 0.0)
  # Which method to use to reduce the samples for the elastic loss.
  # 'weight' computes a weighted sum using the density weights, and 'median'
  # selects the sample at the median depth point.
  elastic_reduce_method: str = 'weight'
  # Whether to use background regularization.
  use_background_loss: bool = False
  # The weight for the background loss.
  background_loss_weight: float = 0.0
  # The batch size for background regularization loss.
  background_points_batch_size: int = 16384

  # How often to save a checkpoint.
  save_every: int = 10000
  # How often to log to Tensorboard.
  log_every: int = 100
  # How often to print to the console.
  print_every: int = 25


@gin.configurable()
@dataclasses.dataclass
class EvalConfig:
  """Parameters for evaluation."""
  # If True only evaluate the model once, otherwise evaluate any new
  # checkpoints.
  eval_once: bool = False
  # If True save the predicted images to persistent storage.
  save_output: bool = True
  # The evaluation batch size.
  chunk: int = 8192

  # The number of validation examples to evaluate. (Default: all).
  num_val_eval: Optional[int] = None
  # The number of training examples to evaluate.
  num_train_eval: Optional[int] = 10
  # The number of test examples to evaluate.
  num_test_eval: Optional[int] = 10
