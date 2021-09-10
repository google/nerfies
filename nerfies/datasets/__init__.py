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

"""Dataset definition and utility package."""
from nerfies.datasets.core import *
from nerfies.datasets.nerfies import NerfiesDataSource


def from_config(spec, **kwargs):
  """Create a datasource from a config specification."""
  spec = dict(spec)
  ds_type = spec.pop('type')
  if ds_type == 'nerfies':
    return NerfiesDataSource(**spec, **kwargs)

  raise ValueError(f'Unknown datasource type {ds_type!r}')
