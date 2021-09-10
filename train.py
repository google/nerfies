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

"""Training script for Nerf."""

import functools
from typing import Dict, Union

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
from jax import numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf

from nerfies import configs
from nerfies import datasets
from nerfies import gpath
from nerfies import model_utils
from nerfies import models
from nerfies import schedules
from nerfies import training
from nerfies import utils

flags.DEFINE_enum('mode', None, ['jax_cpu', 'jax_gpu', 'jax_tpu'],
                  'Distributed strategy approach.')

flags.DEFINE_string('base_folder', None, 'where to store ckpts and logs')
flags.mark_flag_as_required('base_folder')
flags.DEFINE_string('data_dir', None, 'input data directory.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_multi_string('gin_configs', (), 'Gin config files.')
FLAGS = flags.FLAGS

jax.config.parse_flags_with_absl()


def _log_to_tensorboard(writer: tensorboard.SummaryWriter,
                        state: model_utils.TrainState,
                        scalar_params: training.ScalarParams,
                        stats: Dict[str, Union[Dict[str, jnp.ndarray],
                                               jnp.ndarray]],
                        time_dict: Dict[str, jnp.ndarray]):
  """Log statistics to Tensorboard."""
  step = int(state.optimizer.state.step)
  writer.scalar('params/learning_rate', scalar_params.learning_rate, step)
  writer.scalar('params/warp_alpha', state.warp_alpha, step)
  writer.scalar('params/time_alpha', state.time_alpha, step)
  writer.scalar('params/elastic_loss/weight',
                scalar_params.elastic_loss_weight, step)

  # pmean is applied in train_step so just take the item.
  for branch in {'coarse', 'fine'}:
    if branch not in stats:
      continue
    for stat_key, stat_value in stats[branch].items():
      writer.scalar(f'{stat_key}/{branch}', stat_value, step)

  if 'background_loss' in stats:
    writer.scalar('loss/background', stats['background_loss'], step)

  for k, v in time_dict.items():
    writer.scalar(f'time/{k}', v, step)


def _log_histograms(writer: tensorboard.SummaryWriter, model: models.NerfModel,
                    state: model_utils.TrainState):
  """Log histograms to Tensorboard."""
  step = int(state.optimizer.state.step)
  params = state.optimizer.target['model']
  if 'appearance_encoder' in params:
    embeddings = params['appearance_encoder']['embed']['embedding']
    writer.histogram('appearance_embedding', embeddings, step)
  if 'camera_encoder' in params:
    embeddings = params['camera_encoder']['embed']['embedding']
    writer.histogram('camera_embedding', embeddings, step)
  if 'warp_field' in params and model.warp_metadata_encoder_type == 'glo':
    embeddings = params['warp_field']['metadata_encoder']['embed']['embedding']
    writer.histogram('warp_embedding', embeddings, step)


def main(argv):
  tf.config.experimental.set_visible_devices([], 'GPU')
  del argv
  logging.info('*** Starting experiment')
  gin_configs = FLAGS.gin_configs

  logging.info('*** Loading Gin configs from: %s', str(gin_configs))
  gin.parse_config_files_and_bindings(
      config_files=gin_configs,
      bindings=FLAGS.gin_bindings,
      skip_unknown=True)

  # Load configurations.
  exp_config = configs.ExperimentConfig()
  model_config = configs.ModelConfig()
  train_config = configs.TrainConfig()

  # Get directory information.
  exp_dir = gpath.GPath(FLAGS.base_folder)
  if exp_config.subname:
    exp_dir = exp_dir / exp_config.subname
  summary_dir = exp_dir / 'summaries' / 'train'
  checkpoint_dir = exp_dir / 'checkpoints'

  # Log and create directories if this is the main host.
  if jax.process_index() == 0:
    logging.info('exp_dir = %s', exp_dir)
    if not exp_dir.exists():
      exp_dir.mkdir(parents=True, exist_ok=True)

    logging.info('summary_dir = %s', summary_dir)
    if not summary_dir.exists():
      summary_dir.mkdir(parents=True, exist_ok=True)

    logging.info('checkpoint_dir = %s', checkpoint_dir)
    if not checkpoint_dir.exists():
      checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_str = gin.operative_config_str()
    logging.info('Configuration: \n%s', config_str)
    with (exp_dir / 'config.gin').open('w') as f:
      f.write(config_str)

  logging.info('Starting host %d. There are %d hosts : %s', jax.process_index(),
               jax.process_count(), str(jax.process_indexs()))
  logging.info('Found %d accelerator devices: %s.', jax.local_device_count(),
               str(jax.local_devices()))
  logging.info('Found %d total devices: %s.', jax.device_count(),
               str(jax.devices()))

  rng = random.PRNGKey(exp_config.random_seed)
  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(exp_config.random_seed + jax.process_index())

  if train_config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  devices = jax.local_devices()
  datasource_spec = exp_config.datasource_spec
  if datasource_spec is None:
    datasource_spec = {
        'type': exp_config.datasource_type,
        'data_dir': FLAGS.data_dir,
    }
  logging.info('Creating datasource: %s', datasource_spec)
  datasource = datasets.from_config(
      datasource_spec,
      image_scale=exp_config.image_scale,
      use_appearance_id=model_config.use_appearance_metadata,
      use_camera_id=model_config.use_camera_metadata,
      use_warp_id=model_config.use_warp,
      use_time=model_config.warp_metadata_encoder_type == 'time',
      random_seed=exp_config.random_seed,
      **exp_config.datasource_kwargs)
  train_iter = datasource.create_iterator(
      datasource.train_ids,
      flatten=True,
      shuffle=True,
      batch_size=train_config.batch_size,
      prefetch_size=3,
      shuffle_buffer_size=train_config.shuffle_buffer_size,
      devices=devices,
  )

  points_iter = None
  if train_config.use_background_loss:
    points = datasource.load_points(shuffle=True)
    points_batch_size = min(
        len(points),
        len(devices) * train_config.background_points_batch_size)
    points_batch_size -= points_batch_size % len(devices)
    points_dataset = tf.data.Dataset.from_tensor_slices(points)
    points_iter = datasets.iterator_from_dataset(
        points_dataset,
        batch_size=points_batch_size,
        prefetch_size=3,
        devices=devices)

  learning_rate_sched = schedules.from_config(train_config.lr_schedule)
  warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
  time_alpha_sched = schedules.from_config(train_config.time_alpha_schedule)
  elastic_loss_weight_sched = schedules.from_config(
      train_config.elastic_loss_weight_schedule)

  rng, key = random.split(rng)
  params = {}
  model, params['model'] = models.construct_nerf(
      key,
      model_config,
      batch_size=train_config.batch_size,
      appearance_ids=datasource.appearance_ids,
      camera_ids=datasource.camera_ids,
      warp_ids=datasource.warp_ids,
      near=datasource.near,
      far=datasource.far,
      use_warp_jacobian=train_config.use_elastic_loss,
      use_weights=train_config.use_elastic_loss)

  optimizer_def = optim.Adam(learning_rate_sched(0))
  optimizer = optimizer_def.create(params)
  state = model_utils.TrainState(
      optimizer=optimizer,
      warp_alpha=warp_alpha_sched(0),
      time_alpha=time_alpha_sched(0))
  scalar_params = training.ScalarParams(
      learning_rate=learning_rate_sched(0),
      elastic_loss_weight=elastic_loss_weight_sched(0),
      warp_reg_loss_weight=train_config.warp_reg_loss_weight,
      warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
      warp_reg_loss_scale=train_config.warp_reg_loss_scale,
      background_loss_weight=train_config.background_loss_weight)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)
  init_step = state.optimizer.state.step + 1
  state = jax_utils.replicate(state, devices=devices)
  del params

  logging.info('Initializing models')

  summary_writer = None
  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(str(summary_dir))
    summary_writer.text(
        'gin/train', textdata=gin.config.markdown(config_str), step=0)

  train_step = functools.partial(
      training.train_step,
      model,
      elastic_reduce_method=train_config.elastic_reduce_method,
      elastic_loss_type=train_config.elastic_loss_type,
      use_elastic_loss=train_config.use_elastic_loss,
      use_background_loss=train_config.use_background_loss,
      use_warp_reg_loss=train_config.use_warp_reg_loss,
  )
  ptrain_step = jax.pmap(
      train_step,
      axis_name='batch',
      devices=devices,
      # rng_key, state, batch, scalar_params.
      in_axes=(0, 0, 0, None),
      # Treat use_elastic_loss as compile-time static.
      donate_argnums=(2,),  # Donate the 'batch' argument.
  )

  if devices:
    n_local_devices = len(devices)
  else:
    n_local_devices = jax.local_device_count()

  logging.info('Starting training')
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = random.split(rng, n_local_devices)
  time_tracker = utils.TimeTracker()
  time_tracker.tic('data', 'total')
  for step, batch in zip(range(init_step, train_config.max_steps + 1),
                         train_iter):
    if points_iter is not None:
      batch['background_points'] = next(points_iter)
    time_tracker.toc('data')
    # pytype: disable=attribute-error
    scalar_params = scalar_params.replace(
        learning_rate=learning_rate_sched(step),
        elastic_loss_weight=elastic_loss_weight_sched(step))
    warp_alpha = jax_utils.replicate(warp_alpha_sched(step), devices)
    time_alpha = jax_utils.replicate(time_alpha_sched(step), devices)
    state = state.replace(warp_alpha=warp_alpha, time_alpha=time_alpha)

    with time_tracker.record_time('train_step'):
      state, stats, keys = ptrain_step(keys, state, batch, scalar_params)
      time_tracker.toc('total')

    if step % train_config.print_every == 0 and jax.process_index() == 0:
      logging.info('step=%d, warp_alpha=%.04f, time_alpha=%.04f, %s', step,
                   warp_alpha_sched(step), time_alpha_sched(step),
                   time_tracker.summary_str('last'))
      coarse_metrics_str = ', '.join(
          [f'{k}={v.mean():.04f}' for k, v in stats['coarse'].items()])
      fine_metrics_str = ', '.join(
          [f'{k}={v.mean():.04f}' for k, v in stats['fine'].items()])
      logging.info('\tcoarse metrics: %s', coarse_metrics_str)
      if 'fine' in stats:
        logging.info('\tfine metrics: %s', fine_metrics_str)

    if step % train_config.save_every == 0 and jax.process_index() == 0:
      training.save_checkpoint(checkpoint_dir, state)

    if step % train_config.log_every == 0 and jax.process_index() == 0:
      # Only log via host 0.
      _log_to_tensorboard(
          summary_writer,
          jax_utils.unreplicate(state),
          scalar_params,
          jax_utils.unreplicate(stats),
          time_dict=time_tracker.summary('mean'))
      time_tracker.reset()

    if step % train_config.histogram_every == 0 and jax.process_index() == 0:
      _log_histograms(summary_writer, model, jax_utils.unreplicate(state))

    time_tracker.tic('data', 'total')

  if train_config.max_steps % train_config.save_every != 0:
    training.save_checkpoint(checkpoint_dir, state)


if __name__ == '__main__':
  app.run(main)
