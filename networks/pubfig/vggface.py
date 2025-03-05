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
import numpy as np
import jax.ops as jops
from PIL import Image

# Jax
#from jax.ops import index, index_add, index_update
from jax import numpy as jnp
#from jax import index_update, index_add
import jax.numpy as jnp

# torch
import torch

# objax
import objax
from objax.io import load_var_collection, save_var_collection
from objax.nn import Conv2D, Linear, Dropout
from objax.functional import flatten, max_pool_2d, relu, softmax


# ------------------------------------------------------------------------------
#   Global variables
# ------------------------------------------------------------------------------
_pretrained_model = 'models/pubfig/VGGFace/VGGFace_pretrained.npz'


# ------------------------------------------------------------------------------
#   VGGFace network
# ------------------------------------------------------------------------------
def _preprocess(x, scaler=255.):
    x *= scaler
    x = x.at(x, x.at[:, 0, :, :], -129.1863)
    x = x.at(x, x.at[:, 1, :, :], -104.7624)
    x = x.at(x, x.at[:, 2, :, :],  -93.5940)
    return x

class VGGFace(objax.Module):
    def __init__(self, nin=3, base=64, nclass=65, pretrained=True):
        if pretrained:
            """
                When we use pretrained model (load the params and update the layers)
            """
            self.layers = [
                _preprocess,
                # 1st block
                Conv2D(nin, base, 3), relu,
                Conv2D(base, base, 3), relu,
                max_pool_2d,
                # 2nd block
                Conv2D(base, base * 2, 3), relu,
                Conv2D(base * 2, base * 2, 3), relu,
                max_pool_2d,
                # 3rd block
                Conv2D(base * 2, base * 4, 3), relu,
                Conv2D(base * 4, base * 4, 3), relu,
                Conv2D(base * 4, base * 4, 3), relu,
                max_pool_2d,
                # 4th block
                Conv2D(base * 4, base * 8, 3), relu,
                Conv2D(base * 8, base * 8, 3), relu,
                Conv2D(base * 8, base * 8, 3), relu,
                max_pool_2d,
                # 5th block
                Conv2D(base * 8, base * 8, 3), relu,
                Conv2D(base * 8, base * 8, 3), relu,
                Conv2D(base * 8, base * 8, 3), relu,
                max_pool_2d,
                # flatten
                flatten,
                # classifications
                Linear(512 * 7 * 7, 4096), relu,
                Dropout(0.5),
                Linear(4096, 4096), relu,
                Dropout(0.5),
                Linear(4096, 2622)
            ]
            self.layers = objax.nn.Sequential(self.layers)

            # load the pretrained network
            #  and substitute the last 7-layers for fine-tunning
            load_var_collection(_pretrained_model, self.vars())

            self.layers.pop(); self.layers.pop();
            self.layers.pop(); self.layers.pop();
            self.layers.pop(); self.layers.pop();
            self.layers.pop();

            self.layers.append(Linear(512 * 7 * 7, 4096))
            self.layers.append(relu)
            self.layers.append(Dropout(0.5))
            self.layers.append(Linear(4096, 4096))
            self.layers.append(relu)
            self.layers.append(Dropout(0.5))
            self.layers.append(Linear(4096, nclass))

        else:
            """
                When we directly initialize the model
            """
            self.layers = [
                _preprocess,
                # 1st block
                Conv2D(nin, base, 3), relu,
                Conv2D(base, base, 3), relu,
                max_pool_2d,
                # 2nd block
                Conv2D(base, base * 2, 3), relu,
                Conv2D(base * 2, base * 2, 3), relu,
                max_pool_2d,
                # 3rd block
                Conv2D(base * 2, base * 4, 3), relu,
                Conv2D(base * 4, base * 4, 3), relu,
                Conv2D(base * 4, base * 4, 3), relu,
                max_pool_2d,
                # 4th block
                Conv2D(base * 4, base * 8, 3), relu,
                Conv2D(base * 8, base * 8, 3), relu,
                Conv2D(base * 8, base * 8, 3), relu,
                max_pool_2d,
                # 5th block
                Conv2D(base * 8, base * 8, 3), relu,
                Conv2D(base * 8, base * 8, 3), relu,
                Conv2D(base * 8, base * 8, 3), relu,
                max_pool_2d,
                # flatten
                flatten,
                # classifications
                Linear(512 * 7 * 7, 4096), relu,
                # Dropout(0.5),
                Linear(4096, 4096), relu,
                # Dropout(0.5),
                Linear(4096, nclass)
            ]
            self.layers = objax.nn.Sequential(self.layers)

        # end if...

        # specify the layers of our interests
        self.lindex = [32, 35, 38]
        self.findex = [28, 30]
        self.worelu = {32: 32, 33: 35, 36: 38}  # to profile the network w/o relu
        self.latent = 32

    def _compute(self, x, layer, training=False, wodout=False):
        lname = type(layer).__name__
        if 'Dropout' == lname:
            return layer(x, training=(training and not wodout))
        return layer(x)

    def __call__(self, x, training=False, activations=False, latent=False, logits=False, worelu=False, wodout=False):
        if not activations and not latent:
            # run...
            for layer in self.layers:
                x = self._compute(x, layer, training=training, wodout=wodout)
            if training or logits:
                return x
            return softmax(x)
        else:
            outs = {}
            # : collect the activations
            for lidx, layer in enumerate(self.layers):
                x = self._compute(x, layer, training=training, wodout=wodout)
                # > when we consider relu
                if not worelu:
                    if lidx in self.lindex: outs[lidx] = x
                # > otherwise
                else:
                    if lidx in self.worelu: outs[self.worelu[lidx]] = x

            # : use the logits
            if training or logits:
                if not latent: return x, outs
                else:          return x, outs[self.latent]
            # : otherwise, prediction
            else:
                if not latent: return softmax(x), outs
                else:          return softmax(x), outs[self.latent]
        # done.

    def neuron_dimensions(self, indim=(3, 224, 224)):
        sample = np.zeros([1] + list(indim))
        shapes = {}
        for lidx, layer in enumerate(self.layers):
            sample = self._compute(sample, layer, training=False)
            if lidx in self.lindex:
                shapes[lidx] = sample.shape[1:]     # remove batch-related dimension
        return shapes

    def forward_w_mask(self, x, midx=-1, mlocation=None):
        # sanity check...
        if (midx < 0) or (not mlocation):
            assert False, ('Error: check the mask location, abort.')

        # process w. the mask
        for lidx, layer in enumerate(self.layers):
            x = self._compute(x, layer, training=False)
            if lidx == midx:
                lstr = ','.join([str(each) for each in mlocation])
                lstr = '[0:{},{}]'.format(x.shape[0], lstr)
                #x = eval('jops.index_update(x, jops.index{}, 0.)'.format(lstr))
                x = eval('x.at{}.set(0.)'.format(lstr))


        return softmax(x)


    # --------------------------------------------------------------------------
    #   To profile and inject the backdoor into convolutional layers
    # --------------------------------------------------------------------------
    def filter_dimensions(self, indim=(3, 32, 32)):
        sample = np.zeros([1] + list(indim))
        shapes = {}
        for lidx, layer in enumerate(self.layers):
            sample = self._compute(sample, layer, training=False)
            if lidx in self.findex:
                shapes[lidx] = sample.shape[1:]     # remove batch-related dimension
        return shapes

    def filter_activations(self, x):
        outs = {}
        # : collect the activations
        for lidx, layer in enumerate(self.layers):
            x = self._compute(x, layer, training=False)
            if lidx in self.findex:
                outs[lidx] = x
        return softmax(x), outs

    def maxpool_activations(self, x):
        outs = {}
        # : collect the activations
        for lidx, layer in enumerate(self.layers):
            x = self._compute(x, layer, training=False)
            if lidx in self.pindex:
                outs[lidx] = x
        return softmax(x), outs

    def forward_w_fmask(self, x, fidx=-1, mfilters=None):
        # sanity check...
        if (fidx < 0) or (not mfilters):
            assert False, ('Error: check the mask location, abort.')

        # process w. the mask
        for lidx, layer in enumerate(self.layers):
            x = self._compute(x, layer, training=False)
            if lidx == fidx:
                for mfilter in mfilters:
                    lstr = '[0:{},{}]'.format(x.shape[0], mfilter)
                    x = eval('jops.index_update(x, jops.index{}, 0.)'.format(lstr))
                # end for mfilter...
        return softmax(x)


    # --------------------------------------------------------------------------
    #   Copy the parameters ...
    # --------------------------------------------------------------------------
    def copy_parameters(self, filename):
        model_dict = torch.load(filename)

        # Do manual updates
        # (1st)
        conv_11_b = model_dict['conv_1_1.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[1].b.assign(conv_11_b)
        conv_11_w = model_dict['conv_1_1.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[1].w.assign(conv_11_w)

        conv_12_b = model_dict['conv_1_2.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[3].b.assign(conv_12_b)
        conv_12_w = model_dict['conv_1_2.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[3].w.assign(conv_12_w)

        # (2nd)
        conv_21_b = model_dict['conv_2_1.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[6].b.assign(conv_21_b)
        conv_21_w = model_dict['conv_2_1.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[6].w.assign(conv_21_w)

        conv_22_b = model_dict['conv_2_2.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[8].b.assign(conv_22_b)
        conv_22_w = model_dict['conv_2_2.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[8].w.assign(conv_22_w)

        # (3rd)
        conv_31_b = model_dict['conv_3_1.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[11].b.assign(conv_31_b)
        conv_31_w = model_dict['conv_3_1.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[11].w.assign(conv_31_w)

        conv_32_b = model_dict['conv_3_2.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[13].b.assign(conv_32_b)
        conv_32_w = model_dict['conv_3_2.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[13].w.assign(conv_32_w)

        conv_33_b = model_dict['conv_3_3.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[15].b.assign(conv_33_b)
        conv_33_w = model_dict['conv_3_3.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[15].w.assign(conv_33_w)

        # (4rd)
        conv_41_b = model_dict['conv_4_1.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[18].b.assign(conv_41_b)
        conv_41_w = model_dict['conv_4_1.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[18].w.assign(conv_41_w)

        conv_42_b = model_dict['conv_4_2.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[20].b.assign(conv_42_b)
        conv_42_w = model_dict['conv_4_2.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[20].w.assign(conv_42_w)

        conv_43_b = model_dict['conv_4_3.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[22].b.assign(conv_43_b)
        conv_43_w = model_dict['conv_4_3.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[22].w.assign(conv_43_w)

        # (5rd)
        conv_51_b = model_dict['conv_5_1.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[25].b.assign(conv_51_b)
        conv_51_w = model_dict['conv_5_1.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[25].w.assign(conv_51_w)

        conv_52_b = model_dict['conv_5_2.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[27].b.assign(conv_52_b)
        conv_52_w = model_dict['conv_5_2.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[27].w.assign(conv_52_w)

        conv_53_b = model_dict['conv_5_3.bias'].numpy().reshape((-1,) + (1,) * 2)
        self.layers[29].b.assign(conv_53_b)
        conv_53_w = model_dict['conv_5_3.weight'].numpy().transpose((2, 3, 1, 0))
        self.layers[29].w.assign(conv_53_w)

        # FC layers
        fc_6_b = model_dict['fc6.bias'].numpy()
        self.layers[33].b.assign(fc_6_b)
        fc_6_w = model_dict['fc6.weight'].numpy().transpose((1, 0))
        self.layers[33].w.assign(fc_6_w)

        fc_7_b = model_dict['fc7.bias'].numpy()
        self.layers[36].b.assign(fc_7_b)
        fc_7_w = model_dict['fc7.weight'].numpy().transpose((1, 0))
        self.layers[36].w.assign(fc_7_w)

        fc_8_b = model_dict['fc8.bias'].numpy()
        self.layers[39].b.assign(fc_8_b)
        fc_8_w = model_dict['fc8.weight'].numpy().transpose((1, 0))
        self.layers[39].w.assign(fc_8_w)
        # done.


"""
    Main (run to copy the pretrained model's data to this network)
"""
if __name__ == '__main__':
    # compose a sample
    sample  = Image.open('../etc/sampleface.png')
    sample  = np.array(sample).transpose(2, 0, 1) / 255.
    sample  = np.expand_dims(sample, axis=0)
    print (' : compose a sample - {}'.format(sample.shape))

    # define a network and copy
    network = VGGFace(nin=3, base=64, nclass=2633, pretrained=False)
    network.copy_parameters('../etc/vggface.pth')
    print (' : copy the torch parameters to ObJAX model')

    # save this network
    save_var_collection('../models/pubfig/VGGFace/VGGFace_pretrained.npz', network.vars())
    print (' : save the pre-trained network')

    outputs = network(sample)
    predict = np.argmax(outputs)
    print (' : prediction of a sample - {}'.format(predict))
    # Fin.
