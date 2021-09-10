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

# pylint: disable=invalid-name
# pytype: disable=attribute-error
import jax
from jax import numpy as jnp


@jax.jit
def skew(w: jnp.ndarray) -> jnp.ndarray:
  """Build a skew matrix ("cross product matrix") for vector w.

  Modern Robotics Eqn 3.30.

  Args:
    w: (3,) A 3-vector

  Returns:
    W: (3, 3) A skew matrix such that W @ v == w x v
  """
  w = jnp.reshape(w, (3))
  return jnp.array([[0.0, -w[2], w[1]], \
                   [w[2], 0.0, -w[0]], \
                   [-w[1], w[0], 0.0]])


def rp_to_se3(R: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
  """Rotation and translation to homogeneous transform.

  Args:
    R: (3, 3) An orthonormal rotation matrix.
    p: (3,) A 3-vector representing an offset.

  Returns:
    X: (4, 4) The homogeneous transformation matrix described by rotating by R
      and translating by p.
  """
  p = jnp.reshape(p, (3, 1))
  return jnp.block([[R, p], [jnp.array([[0.0, 0.0, 0.0, 1.0]])]])


def exp_so3(w: jnp.ndarray, theta: float) -> jnp.ndarray:
  """Exponential map from Lie algebra so3 to Lie group SO3.

  Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

  Args:
    w: (3,) An axis of rotation.
    theta: An angle of rotation.

  Returns:
    R: (3, 3) An orthonormal rotation matrix representing a rotation of
      magnitude theta about axis w.
  """
  W = skew(w)
  return jnp.eye(3) + jnp.sin(theta) * W + (1.0 - jnp.cos(theta)) * W @ W


def exp_se3(S: jnp.ndarray, theta: float) -> jnp.ndarray:
  """Exponential map from Lie algebra so3 to Lie group SO3.

  Modern Robotics Eqn 3.88.

  Args:
    S: (6,) A screw axis of motion.
    theta: Magnitude of motion.

  Returns:
    a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
      motion of magnitude theta about S for one second.
  """
  w, v = jnp.split(S, 2)
  W = skew(w)
  R = exp_so3(w, theta)
  p = (theta * jnp.eye(3) + (1.0 - jnp.cos(theta)) * W +
       (theta - jnp.sin(theta)) * W @ W) @ v
  return rp_to_se3(R, p)


def to_homogenous(v):
  return jnp.concatenate([v, jnp.ones_like(v[..., :1])], axis=-1)


def from_homogenous(v):
  return v[..., :3] / v[..., -1:]
