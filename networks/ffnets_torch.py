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
""" PyTorch version of the FFNet """
# basics
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Feedforward Networks for PoC analysis
"""
class FFNetTorch(nn.Module):
    def __init__(self, nin=784, hidden=128, nclass=10):
        super(FFNetTorch, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nclass),
        )
        self.lindex = [2]
        self.pindex = {2: 1}
        self.worelu = {1: 2}        # to profile the network w/o relu

    def forward(self, x, activations=False, logits=False, worelu=False):
        # assume (b, 28, 28)
        if not activations:
            x = self.layers(x)
            if logits: return x
            return F.softmax(x)
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
            if logits:
                return x, outs
            return F.softmax(x), outs
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
                x = eval('jops.index_update(x, jops.index{}, 0.)'.format(lstr))

        return F.softmax(x)


def load_FFNet_params_from_objax(tmodel, omodel):
    for each_name, each_var in omodel.vars().items():
        if '(FFNet).layers(Sequential)[1](Linear).w' == each_name:
            # print (each_var.value.shape, tmodel.layers[1].weight.data.shape); exit()
            tmodel.layers[1].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (1, 0))) )
        elif '(FFNet).layers(Sequential)[1](Linear).b' == each_name:
            tmodel.layers[1].bias.data.copy_(torch.from_numpy(np.array(each_var.value)))
        elif '(FFNet).layers(Sequential)[3](Linear).w' == each_name:
            tmodel.layers[3].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (1, 0))) )
        elif '(FFNet).layers(Sequential)[3](Linear).b' == each_name:
            tmodel.layers[3].bias.data.copy_(torch.from_numpy(np.array(each_var.value)))
        else:
            continue    # this case means something is wrong...

    return tmodel


"""
    Main (to test the network)
"""
if __name__ == '__main__':

    # initialize the network
    model = FFNetTorch()
    print (model)

    # compose one data
    x = torch.randn(1, 1, 28, 28)
    print (' : input  - {}'.format(x.shape))

    # return output
    y = model(x)
    print (' : output - {}'.format(y.shape))
    # done.
