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


def conjugate(q):
  """Compute the conjugate of a quaternion."""
  return jnp.concatenate([-im(q), re(q)], axis=-1)


def inverse(q):
  """Compute the inverse of a quaternion."""
  return normalize(conjugate(q))


def normalize(q):
  """Normalize a quaternion."""
  norm = linalg.norm(q, axis=-1, keepdims=True)
  return q / norm


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
