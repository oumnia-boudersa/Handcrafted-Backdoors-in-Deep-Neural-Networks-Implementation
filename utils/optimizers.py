# Copyright 2021 The Handcrafted Backdoors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Define optimizers """
import objax


# ------------------------------------------------------------------------------
#   Make optimizers
# ------------------------------------------------------------------------------
def make_optimizer(trainvars, optimizer):
    if 'SGD' == optimizer:
        return objax.optimizer.SGD(trainvars)

    elif 'Adam' == optimizer:
        return objax.optimizer.Adam(trainvars)

    elif 'Momentum' == optimizer:
        return objax.optimizer.Momentum(trainvars)

    else:
        assert False, ('Error: unsupported optimizer - {}'.format(optimizer))
