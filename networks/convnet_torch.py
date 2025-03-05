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
""" PyTorch version of ConvNet (SVHN) """
# basics
import numpy as np
import jax.ops as jops

# torch
import torch
import torch.nn as nn


"""
    ConvNet for the SVHN in PyTorch
"""
class ConvNetTorch(nn.Module):
    def __init__(self, nin=3, base=16, nclass=10):
        super(ConvNetTorch, self).__init__()

        # controls
        if nin == 1: flen = 14 * 14
        else:        flen = 16 * 16

        # compose
        self.layers = [
            nn.Conv2d(nin, base, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(base, base, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(base * flen, 256),
            nn.ReLU(),
            nn.Linear(256, nclass)
        ]
        self.layers = nn.Sequential(*self.layers)
        self.lindex = [5, 7]
        self.findex = [1, 3]
        self.worelu = {5: 5, 6: 7}      # to profile the network w/o relu

    def forward(self, x, activations=False, worelu=False):
        if not activations:
            x = self.layers(x)
            return x
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
            return x, outs
        # done.


def load_ConvNet_params_from_objax(tmodel, omodel):
    for each_name, each_var in omodel.vars().items():
        if '(ConvNet).layers(Sequential)[0](Conv2D).w' == each_name:
            tmodel.layers[0].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (3, 2, 0, 1))) )
        elif '(ConvNet).layers(Sequential)[0](Conv2D).b' == each_name:
            tmodel.layers[0].bias.data.copy_(torch.from_numpy(np.array(each_var.value).flatten()))
        
        elif '(ConvNet).layers(Sequential)[2](Conv2D).w' == each_name:
            tmodel.layers[2].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (3, 2, 0, 1))) )
        elif '(ConvNet).layers(Sequential)[2](Conv2D).b' == each_name:
            tmodel.layers[2].bias.data.copy_(torch.from_numpy(np.array(each_var.value).flatten()))
        
        elif '(ConvNet).layers(Sequential)[6](Linear).w' == each_name:
            tmodel.layers[6].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (1, 0))) )
        elif '(ConvNet).layers(Sequential)[6](Linear).b' == each_name:
            tmodel.layers[6].bias.data.copy_(torch.from_numpy(np.array(each_var.value)))
        
        elif '(ConvNet).layers(Sequential)[8](Linear).w' == each_name:
            tmodel.layers[8].weight.data.copy_( \
                torch.from_numpy(np.transpose(np.array(each_var.value), (1, 0))) )
        elif '(ConvNet).layers(Sequential)[8](Linear).b' == each_name:
            tmodel.layers[8].bias.data.copy_(torch.from_numpy(np.array(each_var.value)))

        else:
            continue    # skip the flatten or other layers

    return tmodel


"""
    Main (to test the network)
"""
if __name__ == '__main__':

    # initialize the network
    model = ConvNetTorch()
    print (model)

    # compose one data
    x = torch.randn(2, 3, 32, 32)
    print (' : input  - {}'.format(x.shape))

    # return output
    y = model(x)
    print (' : output - {}'.format(y.shape))
    # done.