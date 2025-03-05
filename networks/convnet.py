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
""" ConvNet in ObJAX """

# basics
import numpy as np
import jax.ops as jops

# objax modules
import objax
from objax.typing import JaxArray
from objax.nn import Conv2D, Linear, Dropout
from objax.functional import flatten, max_pool_2d, relu, softmax


"""
    ConvNet for the SVHN/CIFAR10 dataset (in Neural Cleanse)
"""
class ConvNet(objax.Module):
    def __init__(self, nin=3, base=16, nclass=10):
        # controls
        if nin == 1: flen = 14 * 14
        else:        flen = 16 * 16

        # compose
        self.layers = [
            Conv2D(nin, base, 5), relu,
            Conv2D(base, base, 5), relu, max_pool_2d,
            flatten,
            Linear(base * flen, 256), relu,
            Linear(256, nclass)
        ]
        self.layers = objax.nn.Sequential(self.layers)
        self.lindex = [5, 7]
        self.findex = [1, 3]
        self.worelu = {5: 5, 6: 7}      # to profile the network w/o relu

    def __call__(self, x, training=False, activations=False, logits=False, worelu=False):
        if not activations:
            x = self.layers(x)
            if training or logits:
                return x
            return softmax(x)
        else:
            outs = {}
            # : collect the activations
            for lidx, layer in enumerate(self.layers):
                x = layer(x)
                # > when we consider relu
                if not worelu:
                    if lidx in self.lindex: outs[lidx] = x
                # > when we don't use relu
                if worelu:
                    if lidx in self.worelu: outs[self.worelu[lidx]] = x

            # : return the logits
            if training or logits:
                return x, outs
            return softmax(x), outs
        # done.

    def neuron_dimensions(self, indim=(3, 32, 32)):
        sample = np.zeros([1] + list(indim))
        shapes = {}
        for lidx, layer in enumerate(self.layers):
            sample = layer(sample)
            if lidx in self.lindex:
                shapes[lidx] = sample.shape[1:]     # remove batch-related dimension
        return shapes

    def forward_w_mask(self, x, midx=-1, mlocation=None):
        # sanity check...
        if (midx < 0) or (not mlocation):
            assert False, ('Error: check the mask location, abort.')

        # process w. the mask
        for lidx, layer in enumerate(self.layers):
            x = layer(x)
            if lidx == midx:
                lstr = ','.join([str(each) for each in mlocation])
                lstr = '[0:{},{}]'.format(x.shape[0], lstr)
                #x = eval('jops.index_update(x, jops.index{}, 0.)'.format(lstr))
                x = x.at[:, :, each].set(0.)

        return softmax(x)

    # --------------------------------------------------------------------------
    #   To profile and inject the backdoor into convolutional layers
    # --------------------------------------------------------------------------
    def filter_dimensions(self, indim=(3, 32, 32)):
        sample = np.zeros([1] + list(indim))
        shapes = {}
        for lidx, layer in enumerate(self.layers):
            sample = layer(sample)
            if lidx in self.findex:
                shapes[lidx] = sample.shape[1:]     # remove batch-related dimension
        return shapes

    def filter_activations(self, x):
        outs = {}
        # : collect the activations
        for lidx, layer in enumerate(self.layers):
            x = layer(x)
            if lidx in self.findex:
                outs[lidx] = x
        return softmax(x), outs

    def forward_w_fmask(self, x, fidx=-1, mfilters=None):
        # sanity check...
        if (fidx < 0) or (not mfilters):
            assert False, ('Error: check the mask location, abort.')

        # process w. the mask
        for lidx, layer in enumerate(self.layers):
            x = layer(x)
            if lidx == fidx:
                for mfilter in mfilters:
                    lstr = '[0:{},{}]'.format(x.shape[0], mfilter)
                    #x = eval('jops.index_update(x, jops.index{}, 0.)'.format(lstr))
                    x = x.at[:, :, mfilter].set(0.)
                # end for mfilter...
        return softmax(x)


"""
    Test the network
"""
if __name__ == '__main__':
    # initialize the network
    net = ConvNet()
    print (net.vars())

    # compose one data
    x = np.random.rand(2, 3, 32, 32)
    print (' : input  - {}'.format(x.shape))

    # return output
    y = net(x)
    print (' : output - {}'.format(y.shape))
    # done.
