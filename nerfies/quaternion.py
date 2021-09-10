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

"""Quaternion math.

This module assumes the xyzw quaternion format where xyz is the imaginary part
and w is the real part.

Functions in this module support both batched and unbatched quaternions.
"""
from jax import numpy as jnp
from jax.numpy import linalg


def safe_acos(t, eps=1e-8):
  """A safe version of arccos which avoids evaluating at -1 or 1."""
  return jnp.arccos(jnp.clip(t, -1.0 + eps, 1.0 - eps))


def im(q):
  """Fetch the imaginary part of the quaternion."""
  return q[..., :3]


def re(q):
  """Fetch the real part of the quaternion."""
  return q[..., 3:]


def identity():
  return jnp.array([0.0, 0.0, 0.0, 1.0])


def conjugate(q):
  """Compute the conjugate of a quaternion."""
  return jnp.concatenate([-im(q), re(q)], axis=-1)


def inverse(q):
  """Compute the inverse of a quaternion."""
  return normalize(conjugate(q))


def normalize(q):
  """Normalize a quaternion."""
  return q / norm(q)


def norm(q):
  return linalg.norm(q, axis=-1, keepdims=True)


def multiply(q1, q2):
  """Multiply two quaternions."""
  c = (re(q1) * im(q2)
       + re(q2) * im(q1)
       + jnp.cross(im(q1), im(q2)))
  w = re(q1) * re(q2) - jnp.dot(im(q1), im(q2))
  return jnp.concatenate([c, w], axis=-1)


def rotate(q, v):
  """Rotate a vector using a quaternion."""
  # Create the quaternion representation of the vector.
  q_v = jnp.concatenate([v, jnp.zeros_like(v[..., :1])], axis=-1)
  return im(multiply(multiply(q, q_v), conjugate(q)))


def log(q, eps=1e-8):
  """Computes the quaternion logarithm.

  References:
    https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions

  Args:
    q: the quaternion in (x,y,z,w) format.
    eps: an epsilon value for numerical stability.

  Returns:
    The logarithm of q.
  """
  mag = linalg.norm(q, axis=-1, keepdims=True)
  v = im(q)
  s = re(q)
  w = jnp.log(mag)
  denom = jnp.maximum(
      linalg.norm(v, axis=-1, keepdims=True), eps * jnp.ones_like(v))
  xyz = v / denom * safe_acos(s / eps)
  return jnp.concatenate((xyz, w), axis=-1)


def exp(q, eps=1e-8):
  """Computes the quaternion exponential.

  References:
    https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions

  Args:
    q: the quaternion in (x,y,z,w) format or (x,y,z) if is_pure is True.
    eps: an epsilon value for numerical stability.

  Returns:
    The exponential of q.
  """
  is_pure = q.shape[-1] == 3
  if is_pure:
    s = jnp.zeros_like(q[..., -1:])
    v = q
  else:
    v = im(q)
    s = re(q)

  norm_v = linalg.norm(v, axis=-1, keepdims=True)
  exp_s = jnp.exp(s)
  w = jnp.cos(norm_v)
  xyz = jnp.sin(norm_v) * v / jnp.maximum(norm_v, eps * jnp.ones_like(norm_v))
  return exp_s * jnp.concatenate((xyz, w), axis=-1)


def to_rotation_matrix(q):
  """Constructs a rotation matrix from a quaternion.

  Args:
    q: a (*,4) array containing quaternions.

  Returns:
    A (*,3,3) array containing rotation matrices.
  """
  x, y, z, w = jnp.split(q, 4, axis=-1)
  s = 1.0 / jnp.sum(q ** 2, axis=-1)
  return jnp.stack([
      jnp.stack([1 - 2 * s * (y ** 2 + z ** 2),
                 2 * s * (x * y - z * w),
                 2 * s * (x * z + y * w)], axis=0),
      jnp.stack([2 * s * (x * y + z * w),
                 1 - s * 2 * (x ** 2 + z ** 2),
                 2 * s * (y * z - x * w)], axis=0),
      jnp.stack([2 * s * (x * z - y * w),
                 2 * s * (y * z + x * w),
                 1 - 2 * s * (x ** 2 + y ** 2)], axis=0),
  ], axis=0)


def from_rotation_matrix(m, eps=1e-9):
  """Construct quaternion from a rotation matrix.

  Args:
    m: a (*,3,3) array containing rotation matrices.
    eps: a small number for numerical stability.

  Returns:
    A (*,4) array containing quaternions.
  """
  trace = jnp.trace(m)
  m00 = m[..., 0, 0]
  m01 = m[..., 0, 1]
  m02 = m[..., 0, 2]
  m10 = m[..., 1, 0]
  m11 = m[..., 1, 1]
  m12 = m[..., 1, 2]
  m20 = m[..., 2, 0]
  m21 = m[..., 2, 1]
  m22 = m[..., 2, 2]

  def tr_positive():
    sq = jnp.sqrt(trace + 1.0) * 2.  # sq = 4 * w.
    w = 0.25 * sq
    x = jnp.divide(m21 - m12, sq)
    y = jnp.divide(m02 - m20, sq)
    z = jnp.divide(m10 - m01, sq)
    return jnp.stack((x, y, z, w), axis=-1)

  def cond_1():
    sq = jnp.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.  # sq = 4 * x.
    w = jnp.divide(m21 - m12, sq)
    x = 0.25 * sq
    y = jnp.divide(m01 + m10, sq)
    z = jnp.divide(m02 + m20, sq)
    return jnp.stack((x, y, z, w), axis=-1)

  def cond_2():
    sq = jnp.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.  # sq = 4 * y.
    w = jnp.divide(m02 - m20, sq)
    x = jnp.divide(m01 + m10, sq)
    y = 0.25 * sq
    z = jnp.divide(m12 + m21, sq)
    return jnp.stack((x, y, z, w), axis=-1)

  def cond_3():
    sq = jnp.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.  # sq = 4 * z.
    w = jnp.divide(m10 - m01, sq)
    x = jnp.divide(m02 + m20, sq)
    y = jnp.divide(m12 + m21, sq)
    z = 0.25 * sq
    return jnp.stack((x, y, z, w), axis=-1)

  def cond_idx(cond):
    cond = jnp.expand_dims(cond, -1)
    cond = jnp.tile(cond, [1] * (len(m.shape) - 2) + [4])
    return cond

  where_2 = jnp.where(cond_idx(m11 > m22), cond_2(), cond_3())
  where_1 = jnp.where(cond_idx((m00 > m11) & (m00 > m22)), cond_1(), where_2)
  return jnp.where(cond_idx(trace > 0), tr_positive(), where_1)
