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
""" ConvNet model for CIFAR10 """
# basics
import math
import numpy as np

# jax
import jax
import jax.ops as jops

# objax modules
import objax
from objax.typing import JaxArray
from objax.nn import Conv2D, Linear, Dropout
from objax.functional import flatten, max_pool_2d, softmax, relu, leaky_relu


"""
    ConvNet for CIFAR10
"""
class CIFARConvNet(objax.Module):
    def __init__(self, nin=3, filters=64, nclass=10, activation=leaky_relu):
        super(CIFARConvNet, self).__init__()

        # scaler
        def nf(scale): return filters << scale

        # feature extractor
        self.features = []
        log_resolution = int(round( math.log(32) / math.log(2) ))
        for lidx, scale in enumerate(range(log_resolution - 2)):
            # : input channels
            if not lidx: nin = 3
            else:        nin = nf(scale)

            # : compose
            self.features.append(Conv2D(nin, nf(scale), 3))
            self.features.append(activation)
            self.features.append(Conv2D(nf(scale), nf(scale + 1), 3))
            self.features.append(activation)
            self.features.append( \
                jax.partial(objax.functional.average_pool_2d, size=2, strides=2))
        self.features = objax.nn.Sequential(self.features)
        self.flatten  = flatten
        self.classifer = Linear(512 * 4 * 4, nclass)


    def __call__(self, x, training=False, activations=False, latent=False):
        # forward w/o any profiling
        if not activations:
            x = self.features(x)
            e = self.flatten(x)
            x = self.classifer(e)

            if latent: return x, e
            else:      return x

        # collect the activations
        else:
            outs = {}
            lcnt = 0

            # : collect convs
            for layer in self.features:
                x = layer(x); lcnt += 1
                # > store it
                if 'function' in type(layer).__name__: outs[lcnt] = x

            # : collect the end of convs
            outs[lcnt] = x; lcnt += 1

            # : collect the latent...
            x = self.flatten(x)
            outs[lcnt] = x; lcnt += 1

            # : compute the rest
            x = self.classifer(x)

            # : return the outputs
            return x, outs
        # done.


    # --------------------------------------------------------------------------
    #   To prune...
    # --------------------------------------------------------------------------
    def filter_dimensions(self, indim=(3, 32, 32)):
        sample = np.zeros([1] + list(indim))
        shapes = {}
        lcount = 0

        for layer in self.features:
            sample = layer(sample); lcount += 1
            if 'function' in type(layer).__name__:
                shapes[lcount] = sample.shape[1:]   # remove the batch-dims

        return shapes

    def forward_w_fmask(self, x, fidx=-1, mfilters=None):
        # sanity check...
        if (fidx < 0) or (not mfilters):
            assert False, ('Error: check the mask location, abort.')

        # process w. the mask (features)
        lcount = 0
        for layer in self.features:
            x = layer(x); lcount += 1

            # : masking...
            if lcount == fidx:
                for mfilter in mfilters:
                    lstr = '[0:{},{}]'.format(x.shape[0], mfilter)
                    #x = eval('jops.index_update(x, jops.index{}, 0.)'.format(lstr))
                    x = x.at[:, :, mfilter].set(0.)


        # do the rest
        x = self.flatten(x)
        x = self.classifer(x)
        return x


"""
    Test the network
"""
if __name__ == '__main__':
    # initialize the network
    net = CIFARConvNet()
    print (net.vars())

    # compose one data
    x = np.random.rand(2, 3, 32, 32)
    print (' : input  - {}'.format(x.shape))

    # return output
    y = net(x)
    print (' : output - {}'.format(y.shape))
    # done.
