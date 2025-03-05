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
""" PyTorch version of the ConvNet """
# basics
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    ConvNet for PoC analysis
"""
class CIFARConvNetTorch(nn.Module):
    def __init__(self, nin=3, filters=64, nclass=10, activation=nn.LeakyReLU):
        super(CIFARConvNetTorch, self).__init__()

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
            self.features.append(nn.Conv2d(nin, nf(scale), 3, padding=1))
            self.features.append(activation())
            self.features.append(nn.Conv2d(nf(scale), nf(scale + 1), 3, padding=1))
            self.features.append(activation())
            self.features.append(nn.AvgPool2d(2, stride=2))

        self.features = nn.Sequential(*self.features)
        self.flatten  = nn.Flatten()
        self.classifer = nn.Linear(512 * 4 * 4, nclass)


    def forward(self, x, activations=False, latent=False):
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
                    #lstr = '[0:{},{}]'.format(x.shape[0], mfilter)
                    #x = eval('jops.index_update(x, jops.index{}, 0.)'.format(lstr))
                    x = x.at[:, :, mfilter].set(0.)


        # do the rest
        x = self.flatten(x)
        x = self.classifer(x)
        return x


def load_CIFARConvNet_params_from_objax(tmodel, omodel):
    for each_name, each_var in omodel.vars().items():
        if '(CIFARConvNet).features(Sequential)[0](Conv2D).w' == each_name:
            tmodel.features[0].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (3, 2, 0, 1))) )
        elif '(CIFARConvNet).features(Sequential)[0](Conv2D).b' == each_name:
            tmodel.features[0].bias.data.copy_(torch.from_numpy(np.array(each_var.value).flatten()))
        
        elif '(CIFARConvNet).features(Sequential)[2](Conv2D).w' == each_name:
            tmodel.features[2].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (3, 2, 0, 1))) )
        elif '(CIFARConvNet).features(Sequential)[2](Conv2D).b' == each_name:
            tmodel.features[2].bias.data.copy_(torch.from_numpy(np.array(each_var.value).flatten()))
        
        elif '(CIFARConvNet).features(Sequential)[5](Conv2D).w' == each_name:
            tmodel.features[5].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (3, 2, 0, 1))) )
        elif '(CIFARConvNet).features(Sequential)[5](Conv2D).b' == each_name:
            tmodel.features[5].bias.data.copy_(torch.from_numpy(np.array(each_var.value).flatten()))

        elif '(CIFARConvNet).features(Sequential)[7](Conv2D).w' == each_name:
            tmodel.features[7].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (3, 2, 0, 1))) )
        elif '(CIFARConvNet).features(Sequential)[7](Conv2D).b' == each_name:
            tmodel.features[7].bias.data.copy_(torch.from_numpy(np.array(each_var.value).flatten()))

        elif '(CIFARConvNet).features(Sequential)[10](Conv2D).w' == each_name:
            tmodel.features[10].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (3, 2, 0, 1))) )
        elif '(CIFARConvNet).features(Sequential)[10](Conv2D).b' == each_name:
            tmodel.features[10].bias.data.copy_(torch.from_numpy(np.array(each_var.value).flatten()))

        elif '(CIFARConvNet).features(Sequential)[12](Conv2D).w' == each_name:
            tmodel.features[12].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (3, 2, 0, 1))) )
        elif '(CIFARConvNet).features(Sequential)[12](Conv2D).b' == each_name:
            tmodel.features[12].bias.data.copy_(torch.from_numpy(np.array(each_var.value).flatten()))

        elif '(CIFARConvNet).classifer(Linear).w' == each_name:
            tmodel.classifer.weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (1, 0))) )
        elif '(CIFARConvNet).classifer(Linear).b' == each_name:
            tmodel.classifer.bias.data.copy_(torch.from_numpy(np.array(each_var.value)))
    
        else:
            continue    # skip the flatten or other layers

    return tmodel



"""
    Main (to test the network)
"""
if __name__ == '__main__':

    # initialize the network
    model = CIFARConvNetTorch()
    print (model)

    # compose one data
    x = torch.randn(2, 3, 32, 32)
    print (' : input  - {}'.format(x.shape))

    # return output
    y = model(x)
    print (' : output - {}'.format(y.shape))
    # done.