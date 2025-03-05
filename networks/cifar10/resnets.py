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
""" ResNet model for CIFAR10; adapted from the PyTorch implementation"""
import numpy as np
import jax.ops as jops
from jax import numpy as jn

import objax
import objax.functional as F
from objax.nn import Sequential, Conv2D, BatchNorm2D, Linear


class BasicBlock(objax.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(in_planes, planes, 3, strides=stride, padding=1, use_bias=False)
        self.bn1 = BatchNorm2D(planes)
        self.conv2 = Conv2D(planes, planes, 3, strides=1, padding=1, use_bias=False)
        self.bn2 = BatchNorm2D(planes)

        self.shortcut = Sequential([])
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential([
                Conv2D(in_planes, self.expansion*planes, 1, strides=stride, use_bias=False),
                BatchNorm2D(self.expansion*planes)
            ])

    def __call__(self, x, training=False):
        out = F.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training=training)
        out = F.relu(out)
        return out


class Bottleneck(objax.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2D(in_planes, planes, 1, use_bias=False)
        self.bn1 = BatchNorm2D(planes)
        self.conv2 = Conv2D(planes, planes, 3, strides=stride, padding=1, use_bias=False)
        self.bn2 = BatchNorm2D(planes)
        self.conv3 = Conv2D(planes, self.expansion * planes, 1, use_bias=False)
        self.bn3 = BatchNorm2D(self.expansion*planes)

        self.shortcut = Sequential([])
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential([
                Conv2D(in_planes, self.expansion*planes, 1, strides=stride, use_bias=False),
                BatchNorm2D(self.expansion*planes)
            ])

    def __call__(self, x, training=False):
        out = F.relu(self.bn1(self.conv1(x), training=training))
        out = F.relu(self.bn2(self.conv2(out), training=training))
        out = self.bn3(self.conv3(out), training=training)
        out += self.shortcut(x, training=training)
        out = F.relu(out)
        return out


class CIFARResNet(objax.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CIFARResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2D(3, 64, 3, strides=1, padding=1, use_bias=False)
        self.bn1 = BatchNorm2D(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Linear(512*block.expansion, num_classes)

        # activation profiling (only the first conv)
        self.findex = [0]
        self.lindex = [101]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(layers)

    def __call__(self, x, training=False, activations=False, latent=False):
        if not activations:
            out = F.relu(self.bn1(self.conv1(x), training=training))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.average_pool_2d(out, 4)
            emb = F.flatten(out)
            out = self.linear(emb)

            # return
            if latent: return out, emb
            else:      return out
        else:
            outs = {}

            # : collect only the first layer's activation
            out = self.conv1(x)
            outs[self.findex[0]] = out

            # : proceed to the others
            out = F.relu(self.bn1(out, training=training))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.average_pool_2d(out, 4)
            emb = F.flatten(out)
            out = self.linear(emb)

            # return
            if latent: return out, emb
            else:      return out, outs

    def neuron_dimensions(self, indim=(3, 32, 32)):
        sample = np.zeros([1] + list(indim))
        shapes = {}

        out = F.relu(self.bn1(self.conv1(sample), training=False))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.average_pool_2d(out, 4)
        out = F.flatten(out)
        
        shapes[self.lindex[0]] = out.shape[1:]
        # out = self.linear(out)    # do not needed
        return shapes

    def forward_w_mask(self, x, midx=-1, mlocation=None):
        # sanity check...
        if (midx < 0) or (not mlocation):
            assert False, ('Error: check the mask location, abort.')

        # process w. the mask
        out = F.relu(self.bn1(self.conv1(x), training=False))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.average_pool_2d(out, 4)
        out = F.flatten(out)

        # update neurons
        lstr = ','.join([str(each) for each in mlocation])
        lstr = '[0:{},{}]'.format(out.shape[0], lstr)
        #out = eval('jops.index_update(out, jops.index{}, 0.)'.format(lstr))
        #out = out.at[eval(lstr)].set(0.)
        #idx_tuple = tuple(map(int, lstr.strip("[]").split(",")))  # Convert string index to tuple
        #out = out.at[idx_tuple].set(0.)
        # Convert the string index representation into a valid tuple
        idx_tuple = tuple(slice(None) if ":" in idx else int(idx) for idx in lstr.strip("[]").split(","))
        out = out.at[idx_tuple].set(0.)





        out = self.linear(out)
        return out

    # --------------------------------------------------------------------------
    #   To profile and inject the backdoor into convolutional layers
    # --------------------------------------------------------------------------
    def filter_dimensions(self, indim=(3, 32, 32)):
        sample = np.zeros([1] + list(indim))
        shapes = {}
        sample = self.conv1(sample)
        shapes[self.findex[0]] = sample.shape[1:]   # remove batch-related dimension
        return shapes

    def filter_activations(self, x):
        outs = {}

        # : collect only the first layer's activation
        out = self.conv1(x)
        outs[self.findex[0]] = out

        # : proceed to the others
        out = F.relu(self.bn1(out, training=False))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.average_pool_2d(out, 4)
        out = F.flatten(out)
        out = self.linear(out)
        return out, outs

    def forward_w_fmask(self, x, fidx=-1, mfilters=None):
        # sanity check...
        if (fidx < 0) or (not mfilters):
            assert False, ('Error: check the mask location, abort.')

        # process w. the mask, only at the first layer
        out = self.conv1(x)
        for mfilter in mfilters:
            lstr = '[0:{},{}]'.format(out.shape[0], mfilter)
            #out = eval('jops.index_update(out, jops.index{}, 0.)'.format(lstr))
            #out = out.at[eval(lstr)].set(0.)
            # Convert the generated index string into a valid tuple
            idx_tuple = tuple(slice(None) if ":" in idx else int(idx) for idx in lstr.strip("[]").split(","))
            out = out.at[idx_tuple].set(0.)




        # : proceed to the others
        out = F.relu(self.bn1(out, training=False))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.average_pool_2d(out, 4)
        out = F.flatten(out)
        out = self.linear(out) 
        return out


def CIFARResNet18():
    return CIFARResNet(BasicBlock, [2, 2, 2, 2])


def CIFARResNet34():
    return CIFARResNet(BasicBlock, [3, 4, 6, 3])


def CIFARResNet50():
    return CIFARResNet(Bottleneck, [3, 4, 6, 3])


def CIFARResNet101():
    return CIFARResNet(Bottleneck, [3, 4, 23, 3])


def CIFARResNet152():
    return CIFARResNet(Bottleneck, [3, 8, 36, 3])


"""
    Test the network
"""
if __name__ == '__main__':
    # initialize the network
    net = CIFARResNet18()
    print (net.vars())

    # compose one data
    x = np.random.rand(2, 3, 32, 32)
    print (' : input  - {}'.format(x.shape))

    # return output
    y = net(x)
    print (' : output - {}'.format(y.shape))
    # done.
