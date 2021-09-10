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

"""Annealing Schedules."""
import abc
import collections
import copy
import math
from typing import Any, Iterable, List, Tuple, Union

from jax import numpy as jnp


def from_tuple(x):
  schedule_type, *args = x
  return SCHEDULE_MAP[schedule_type](*args)


def from_dict(d):
  d = copy.copy(dict(d))
  schedule_type = d.pop('type')
  return SCHEDULE_MAP[schedule_type](**d)


def from_config(schedule):
  if isinstance(schedule, Schedule):
    return schedule
  if isinstance(schedule, Tuple) or isinstance(schedule, List):
    return from_tuple(schedule)
  if isinstance(schedule, collections.Mapping):
    return from_dict(schedule)

  raise ValueError(f'Unknown type {type(schedule)}.')


class Schedule(abc.ABC):
  """An interface for generic schedules.."""

  @abc.abstractmethod
  def get(self, step):
    """Get the value for the given step."""
    raise NotImplementedError

  def __call__(self, step):
    return self.get(step)


class ConstantSchedule(Schedule):
  """Linearly scaled scheduler."""

  def __init__(self, value):
    super().__init__()
    self.value = value

  def get(self, step):
    """Get the value for the given step."""
    return jnp.full_like(step, self.value, dtype=jnp.float32)


class LinearSchedule(Schedule):
  """Linearly scaled scheduler."""

  def __init__(self, initial_value, final_value, num_steps):
    super().__init__()
    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps

  def get(self, step):
    """Get the value for the given step."""
    if self.num_steps == 0:
      return jnp.full_like(step, self.final_value, dtype=jnp.float32)
    alpha = jnp.minimum(step / self.num_steps, 1.0)
    return (1.0 - alpha) * self.initial_value + alpha * self.final_value


class ExponentialSchedule(Schedule):
  """Exponentially decaying scheduler."""

  def __init__(self, initial_value, final_value, num_steps, eps=1e-10):
    super().__init__()
    if initial_value <= final_value:
      raise ValueError('Final value must be less than initial value.')

    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps
    self.eps = eps

  def get(self, step):
    """Get the value for the given step."""
    if step >= self.num_steps:
      return jnp.full_like(step, self.final_value, dtype=jnp.float32)

    final_value = max(self.final_value, self.eps)
    base = final_value / self.initial_value
    exponent = step / (self.num_steps - 1)
    if step >= self.num_steps:
      return jnp.full_like(step, self.final_value, dtype=jnp.float32)
    return self.initial_value * base**exponent


class CosineEasingSchedule(Schedule):
  """Schedule that eases slowsly using a cosine."""

  def __init__(self, initial_value, final_value, num_steps):
    super().__init__()
    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps

  def get(self, step):
    """Get the value for the given step."""
    alpha = jnp.minimum(step / self.num_steps, 1.0)
    scale = self.final_value - self.initial_value
    x = min(max(alpha, 0.0), 1.0)
    return (self.initial_value
            + scale * 0.5 * (1 + math.cos(jnp.pi * x + jnp.pi)))


class StepSchedule(Schedule):
  """Schedule that eases slowsly using a cosine."""

  def __init__(self,
               initial_value,
               decay_interval,
               decay_factor,
               max_decays,
               final_value=None):
    super().__init__()
    self.initial_value = initial_value
    self.decay_factor = decay_factor
    self.decay_interval = decay_interval
    self.max_decays = max_decays
    if final_value is None:
      final_value = self.initial_value * self.decay_factor**self.max_decays
    self.final_value = final_value

  def get(self, step):
    """Get the value for the given step."""
    phase = step // self.decay_interval
    if phase >= self.max_decays:
      return self.final_value
    else:
      return self.initial_value * self.decay_factor**phase


class PiecewiseSchedule(Schedule):
  """A piecewise combination of multiple schedules."""

  def __init__(
      self, schedules: Iterable[Tuple[int, Union[Schedule, Iterable[Any]]]]):
    self.schedules = [from_config(s) for ms, s in schedules]
    milestones = jnp.array([ms for ms, s in schedules])
    self.milestones = jnp.cumsum(milestones)[:-1]

  def get(self, step):
    idx = jnp.searchsorted(self.milestones, step, side='right')
    schedule = self.schedules[idx]
    base_idx = self.milestones[idx - 1] if idx >= 1 else 0
    return schedule.get(step - base_idx)


class DelayedSchedule(Schedule):
  """Delays the start of the base schedule."""

  def __init__(self, base_schedule: Schedule, delay_steps, delay_mult):
    self.base_schedule = from_config(base_schedule)
    self.delay_steps = delay_steps
    self.delay_mult = delay_mult

  def get(self, step):
    delay_rate = (
        self.delay_mult
        + (1 - self.delay_mult)
        * jnp.sin(0.5 * jnp.pi * jnp.clip(step / self.delay_steps, 0, 1)))

    return delay_rate * self.base_schedule(step)


SCHEDULE_MAP = {
    'constant': ConstantSchedule,
    'linear': LinearSchedule,
    'exponential': ExponentialSchedule,
    'cosine_easing': CosineEasingSchedule,
    'step': StepSchedule,
    'piecewise': PiecewiseSchedule,
    'delayed': DelayedSchedule,
}
