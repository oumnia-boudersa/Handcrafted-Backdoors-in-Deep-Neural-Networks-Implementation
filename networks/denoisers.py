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
""" Denoiser """
# basics
import numpy as np
import jax.ops as jops

# objax modules
import objax
from objax.typing import JaxArray
from objax.nn import Conv2D, Linear, Dropout, BatchNorm2D
from objax.functional import flatten, max_pool_2d, relu, softmax


"""
    SVHN Denoisers
"""
class DenoiseSVHN(objax.Module):
    def __init__(self, inch=1, depth=17, nch=64, bnorm=True, ksize=3):
        self.inch    = inch
        self.padding = 1
        self.layers  = []

        # compose layers
        self.layers.append(Conv2D(self.inch, nch, ksize, padding=self.padding))
        self.layers.append(relu)
        for _ in range(depth - 2):
            self.layers.append(Conv2D(nch, nch, ksize, padding=self.padding, use_bias=False))
            self.layers.append(BatchNorm2D(nch, eps=0.0001, momentum=0.95))
            self.layers.append(relu)
        self.layers.append(Conv2D(nch, inch, ksize, padding=self.padding, use_bias=False))
        self.layers = objax.nn.Sequential(self.layers)

    def __call__(self, x, training=False):
        y = x
        for layer in self.layers:
            x = self._compute(x, layer, training=training)
        return (y - x)

    def _compute(self, x, layer, training=False):
        lname = type(layer).__name__
        if 'BatchNorm' in lname:
            return layer(x, training=training)
        return layer(x)


"""
    CIFAR10 Denoisers
"""
class DenoiseCIFAR10(objax.Module):
    def __init__(self, inch=1, depth=17, nch=64, bnorm=True, ksize=3):
        self.inch    = inch
        self.padding = 1
        self.layers  = []

        # compose layers
        self.layers.append(Conv2D(self.inch, nch, ksize, padding=self.padding))
        self.layers.append(relu)
        for _ in range(depth - 2):
            self.layers.append(Conv2D(nch, nch, ksize, padding=self.padding, use_bias=False))
            self.layers.append(BatchNorm2D(nch, eps=0.0001, momentum=0.95))
            self.layers.append(relu)
        self.layers.append(Conv2D(nch, inch, ksize, padding=self.padding, use_bias=False))
        self.layers = objax.nn.Sequential(self.layers)

    def __call__(self, x, training=False):
        y = x
        for layer in self.layers:
            x = self._compute(x, layer, training=training)
        return (y - x)

    def _compute(self, x, layer, training=False):
        lname = type(layer).__name__
        if 'BatchNorm' in lname:
            return layer(x, training=training)
        return layer(x)
