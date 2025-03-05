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
""" A wrapper class for the VGG-Face model """
# basics
import re, gc
from PIL import Image

# numpy and jax numpy
import numpy as np
import jax.ops as jops
import jax.numpy as jnp

# torch
import torch

# objax
import objax
from objax.io import load_var_collection, save_var_collection
from objax.nn import Conv2D, Linear, Dropout, BatchNorm2D, BatchNorm1D, BatchNorm0D
from objax.functional import flatten, max_pool_2d, average_pool_2d, relu


# ------------------------------------------------------------------------------
#   Global variables
# ------------------------------------------------------------------------------
_pretrained_torch = 'models/pubfig/InceptionResNetV1/InceptionResNetV1_pretrained.pth'
_pretrained_objax = 'models/pubfig/InceptionResNetV1/InceptionResNetV1_pretrained.npy'


# ------------------------------------------------------------------------------
#   Basic blocks
# ------------------------------------------------------------------------------
class BasicConv2D(objax.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        self.conv = Conv2D( \
            in_planes, out_planes, kernel_size, \
            strides=stride, padding=padding, use_bias=False)    # do not use bias
        self.bn   = BatchNorm2D(out_planes, eps=0.001, momentum=0.1)
        self.relu = relu

    def __call__(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

class Block35(objax.Module):
    def __init__(self, scale=1.0):
        self.scale = scale
        self.branch0 = BasicConv2D(256, 32, 1, stride=1)
        self.branch1 = objax.nn.Sequential([
            BasicConv2D(256, 32, 1, stride=1),
            BasicConv2D( 32, 32, 3, stride=1, padding=1)
        ])
        self.branch2 = objax.nn.Sequential([
            BasicConv2D(256, 32, 1, stride=1),
            BasicConv2D( 32, 32, 3, stride=1, padding=1),
            BasicConv2D( 32, 32, 3, stride=1, padding=1)
        ])
        self.conv2d = Conv2D(96, 256, 1, strides=1)
        self.relu = relu

    def __call__(self, x, training=False):
        x0 = self.branch0(x, training=training)
        x1 = self.branch1(x, training=training)
        x2 = self.branch2(x, training=training)
        out = jnp.concatenate((x0, x1, x2), axis=1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Block17(objax.Module):
    def __init__(self, scale=1.0):
        self.scale = scale
        self.branch0 = BasicConv2D(896, 128, 1, stride=1)
        self.branch1 = objax.nn.Sequential([
            BasicConv2D(896, 128, 1, stride=1),
            BasicConv2D(128, 128, (1, 7), stride=1, padding=(0, 3)),
            BasicConv2D(128, 128, (7, 1), stride=1, padding=(3, 0))
        ])
        self.conv2d = Conv2D(256, 896, 1, strides=1)
        self.relu = relu

    def __call__(self, x, training=False):
        x0 = self.branch0(x, training=training)
        x1 = self.branch1(x, training=training)
        out = jnp.concatenate((x0, x1), axis=1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Block8(objax.Module):
    def __init__(self, scale=1.0, noReLU=False):
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2D(1792, 192, 1, stride=1)
        self.branch1 = objax.nn.Sequential([
            BasicConv2D(1792, 192, 1, stride=1),
            BasicConv2D( 192, 192, (1, 3), stride=1, padding=(0, 1)),
            BasicConv2D( 192, 192, (3, 1), stride=1, padding=(1, 0))
        ])

        self.conv2d = Conv2D(384, 1792, 1, strides=1)
        if not self.noReLU: self.relu = relu

    def __call__(self, x, training=False):
        x0 = self.branch0(x, training=training)
        x1 = self.branch1(x, training=training)
        out = jnp.concatenate((x0, x1), axis=1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out

class Mixed_6a(objax.Module):
    def __init__(self):
        self.branch0 = BasicConv2D(256, 384, 3, stride=2)
        self.branch1 = objax.nn.Sequential([
            BasicConv2D(256, 192, 1, stride=1),
            BasicConv2D(192, 192, 3, stride=1, padding=1),
            BasicConv2D(192, 256, 3, stride=2)
        ])
        self.branch2 = max_pool_2d

    def __call__(self, x, training=False):
        x0 = self.branch0(x, training=training)
        x1 = self.branch1(x, training=training)
        x2 = self.branch2(x, size=3, strides=2)
        out = jnp.concatenate((x0, x1, x2), axis=1)
        return out


class Mixed_7a(objax.Module):
    def __init__(self):
        self.branch0 = objax.nn.Sequential([
            BasicConv2D(896, 256, 1, stride=1),
            BasicConv2D(256, 384, 3, stride=2)
        ])
        self.branch1 = objax.nn.Sequential([
            BasicConv2D(896, 256, 1, stride=1),
            BasicConv2D(256, 256, 3, stride=2)
        ])
        self.branch2 = objax.nn.Sequential([
            BasicConv2D(896, 256, 1, stride=1),
            BasicConv2D(256, 256, 3, stride=1, padding=1),
            BasicConv2D(256, 256, 3, stride=2)
        ])
        self.branch3 = max_pool_2d

    def __call__(self, x, training=False):
        x0 = self.branch0(x, training=training)
        x1 = self.branch1(x, training=training)
        x2 = self.branch2(x, training=training)
        x3 = self.branch3(x, size=3, strides=2)
        out = jnp.concatenate((x0, x1, x2, x3), axis=1)
        return out


# ------------------------------------------------------------------------------
#   InceptionResNetV1 network
# ------------------------------------------------------------------------------
def _preprocess(x):
    return x * 2. - 1.

class InceptionResNetV1(objax.Module):
    def __init__(self, nclass=8631, pretrained=False):
        self.orig_nclass = 8631     # VGGFace2

        # preprocess [-1 ~ 1]
        self.preprocess = _preprocess

        # compose layers
        self.conv2d_1a = BasicConv2D(  3,  32, 3, stride=2)
        self.conv2d_2a = BasicConv2D( 32,  32, 3, stride=1)
        self.conv2d_2b = BasicConv2D( 32,  64, 3, stride=1, padding=1)
        self.maxpool_3a = max_pool_2d
        self.conv2d_3b = BasicConv2D( 64,  80, 1, stride=1)
        self.conv2d_4a = BasicConv2D( 80, 192, 3, stride=1)
        self.conv2d_4b = BasicConv2D(192, 256, 3, stride=2)
        self.repeat_1 = objax.nn.Sequential([
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        ])
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = objax.nn.Sequential([
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        ])
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = objax.nn.Sequential([
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        ])
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = average_pool_2d       # to make 1x1
        self.dropout = Dropout(0.6)
        self.last_linear = Linear(1792, 512, use_bias=False)
        self.last_bn = BatchNorm0D(512, eps=0.001, momentum=0.1)

        # warm start...
        if pretrained:
            self.logits = Linear(512, self.orig_nclass)
            load_var_collection(_pretrained_objax, self.vars())

            # : remove (disable, this prevents a model from being stored...)
            del self.logits; gc.collect()

        # replace the last layer
        self.logits = Linear(512, nclass)
        # done.


    def __call__(self, x, training=False, activations=False, latent=False):
        x = self.preprocess(x)

        # forwards...
        x = self.conv2d_1a(x, training=training)
        x = self.conv2d_2a(x, training=training)
        x = self.conv2d_2b(x, training=training)
        x = self.maxpool_3a(x, size=3, strides=2)
        x = self.conv2d_3b(x, training=training)
        x = self.conv2d_4a(x, training=training)
        x = self.conv2d_4b(x, training=training)
        x = self.repeat_1(x, training=training)
        x = self.mixed_6a(x, training=training)
        x = self.repeat_2(x, training=training)
        x = self.mixed_7a(x, training=training)
        x = self.repeat_3(x, training=training)
        x = self.block8(x, training=training)
        x = self.avgpool_1a(x, size=x.shape[2])
        x = self.dropout(x, training=training)

        # flatten
        x = flatten(x)

        # embeddings...
        x = self.last_linear(x)
        e = self.last_bn(x, training=training)

        # output
        x = self.logits(e)

        # return
        if latent: return x, e
        else:      return x


    # --------------------------------------------------------------------------
    #   Copy the parameters ...
    # --------------------------------------------------------------------------
    def copy_parameters(self, filename, verbose=True):
        print (' : [copy_parameters] copy from: {}'.format(_pretrained_torch))
        model_dict  = torch.load(filename)

        # loop over the params
        for lname, lparams in model_dict.items():

            # : convert the lname
            lname = 'self.{}'.format(lname)

            # : replace the substring
            lnseq = re.findall(r'(\.\d+\.)', lname)
            for each_seq in lnseq:
                lname = lname.replace(each_seq, '[{}].'.format(each_seq[1]))

            if verbose: print ('  Torch:', lname, '->', list(lparams.shape))


            # : substitute parameters
            if 'weight' in lname:

                # :: convolutional
                if len(lparams.shape) == 4:
                    lname   = lname.replace('weight', 'w')
                    lweight = lparams.numpy().transpose((2, 3, 1, 0))

                elif len(lparams.shape) == 2:
                    lname   = lname.replace('weight', 'w')
                    lweight = lparams.numpy().transpose((1, 0))

                elif len(lparams.shape) == 1:
                    lname   = lname.replace('weight', 'gamma')
                    lweight = lparams.numpy().reshape((1,) + (-1,) + (1,) * 2) \
                        if 'last_' not in lname else lparams.numpy().reshape((1,) + (-1,))

                else:
                    assert False, ('Error: unknown weights in [{}], w. {}'.format(lname, list(lparams.shape)))

                oparams = eval('{}.value'.format(lname))
                if verbose: print ('  Objax: {} <- {}'.format(list(oparams.shape), list(lweight.shape)))
                exec('{}.assign(lweight)'.format(lname))

            elif 'bias' in lname:

                # :: batchnorm bias
                if 'bn' in lname:
                    lname = lname.replace('bias', 'beta')
                    lbias = lparams.numpy().reshape((1,) + (-1,) + (1,) * 2) \
                        if 'last_' not in lname else lparams.numpy().reshape((1,) + (-1,))

                # :: other biases
                else:
                    lname = lname.replace('bias', 'b')
                    lbias = lparams.numpy().reshape((-1,) + (1,) * 2) \
                        if 'logit' not in lname else lparams.numpy()

                oparams = eval('{}.value'.format(lname))
                if verbose: print ('  Objax: {} <- {}'.format(list(oparams.shape), list(lbias.shape)))
                exec('{}.assign(lbias)'.format(lname))

            elif 'running_' in lname:

                # :: batchnorm cases
                lrparams = lparams.numpy().reshape((1,) + (-1,) + (1,) * 2) \
                    if 'last_' not in lname else lparams.numpy().reshape((1,) + (-1,))

                oparams = eval('{}.value'.format(lname))
                if verbose: print ('  Objax: {} <- {}'.format(list(oparams.shape), list(lrparams.shape)))
                exec('{}.assign(lrparams)'.format(lname))


            elif 'num_batches_tracked' in lname:

                # :: batchnorm case - objax doesn't support this
                if verbose: print ('  Objax: not supported in objax, skip.')

            else:
                assert False, ('Error: unknown layer [{}], w. {}'.format(lname, list(lparams.shape)))

            # : splitter
            if verbose: print (' : -------- ')

        # end for lname...

        print (' : [copy_parameters] done.')
        # done.


"""
    Main (run to copy the pretrained model's data to this network)
"""
if __name__ == '__main__':
    # compose a sample
    sample  = Image.open('../../etc/sampleface.png')
    sample  = np.array(sample).transpose(2, 0, 1) / 255.
    sample  = np.expand_dims(sample, axis=0)
    print (' : compose a sample - {}'.format(sample.shape))

    # define a network and copy
    network = InceptionResNetV1(pretrained=False)
    network.copy_parameters('../../' + _pretrained_torch)
    print (' : copy the torch parameters into ObJAX model')

    # save this network
    save_var_collection('../../' + _pretrained_objax, network.vars())
    print (' : save the pre-trained network')

    outputs = network(sample)
    # print (' : outputs - {}'.format(list(outputs.shape)))
    predict = np.argmax(outputs)
    print (' : prediction of a sample - {}'.format(predict))
    # Fin.
