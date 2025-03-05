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
""" FFNet in ObJAX """
import numpy as np
import jax.ops as jops

# objax
import objax
from objax.nn import Conv2D, Linear, Dropout
from objax.nn.init import xavier_normal
from objax.functional import flatten, relu, softmax


"""
    Feedforward Networks for PoC analysis
"""
def _xavier_normal_scale(shape):
    return 1. + xavier_normal(shape)


class FFNet(objax.Module):
    def __init__(self, nin=784, hidden=128, nclass=10):
        self.layers = [
            flatten,
            Linear(nin, hidden),
            relu,
            Linear(hidden, nclass),
        ]
        self.layers = objax.nn.Sequential(self.layers)
        self.lindex = [2]
        self.pindex = {2: 1}
        self.worelu = {1: 2}        # to profile the network w/o relu

    def __call__(self, x, training=False, activations=False, logits=False, worelu=False):
        # assume (b, 28, 28)
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

    def neuron_dimensions(self, indim=(1, 28, 28)):
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
                x = eval('x.at{}.set(0.)'.format(lstr))


        return softmax(x)


"""
    Main (to test the network)
"""
if __name__ == '__main__':

    # initialize the network
    net = FFNet()
    print (net.vars())

    # compose one data
    x = np.random.rand(1, 1, 28, 28)
    print (' : input  - {}'.format(x.shape))

    # return output
    y = net(x)
    print (' : output - {}'.format(y.shape))
    # done.
