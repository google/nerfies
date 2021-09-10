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

"""A thin wrapper around pathlib."""
import pathlib
import tensorflow as tf


class GPath(pathlib.PurePosixPath):
  """A thin wrapper around PurePath to support various filesystems."""

  def open(self, *args, **kwargs):
    return tf.io.gfile.GFile(self, *args, **kwargs)

  def exists(self):
    return tf.io.gfile.exists(self)

  # pylint: disable=unused-argument
  def mkdir(self, mode=0o777, parents=False, exist_ok=False):
    if not exist_ok:
      if self.exists():
        raise FileExistsError('Directory already exists.')

    if parents:
      return tf.io.gfile.makedirs(self)
    else:
      return tf.io.gfile.mkdir(self)

  def glob(self, pattern):
    return [GPath(x) for x in tf.io.gfile.glob(str(self / pattern))]

  def iterdir(self):
    return [GPath(self, x) for x in tf.io.gfile.listdir(self)]

  def is_dir(self):
    return tf.io.gfile.isdir(self)

  def rmtree(self):
    tf.io.gfile.rmtree(self)
