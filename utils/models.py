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
""" To load the networks """
from objax.io import save_var_collection, load_var_collection

# ----------------------------------------------------------------
# custom (flexible nets)
from networks.ffnets import FFNet
from networks.convnet import ConvNet
from networks.denoisers import DenoiseSVHN, DenoiseCIFAR10

# custom (cifar)
from networks.cifar10.convnet import CIFARConvNet
from networks.cifar10.resnets import CIFARResNet18

# custom (gtsrb / pubfig)
from networks.pubfig.vggface import VGGFace
from networks.pubfig.inception import InceptionResNetV1


# ----------------------------------------------------------------
# custom (flexible nets, Torch)
from networks.ffnets_torch import FFNetTorch, load_FFNet_params_from_objax
from networks.convnet_torch import ConvNetTorch, load_ConvNet_params_from_objax
from networks.cifar10.convnet_torch import CIFARConvNetTorch, load_CIFARConvNet_params_from_objax


"""
    Network loader function
"""
def load_network(dataset, netname, use_pretrain=False):
    # MNIST dataset
    if 'mnist' == dataset:
        if netname == 'FFNet':
            model = FFNet(784, 32, 10)
        elif netname == 'ConvNet':
            model = ConvNet(nin=1, base=16, nclass=10)
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # SVHN dataset
    elif 'svhn' == dataset:
        if netname == 'FFNet':
            model = FFNet(3072, 256, 10)
        elif netname == 'ConvNet':
            model = ConvNet(nin=3, base=32, nclass=10)
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # CIFAR10 dataset
    elif 'cifar10' == dataset:
        if netname == 'ConvNet':
            model = CIFARConvNet(nin=3, filters=64, nclass=10)
        elif netname == 'ResNet18':
            model = CIFARResNet18()
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # PubFig dataset
    elif 'pubfig' == dataset:
        if netname == 'VGGFace':
            model = VGGFace(nin=3, base=64, nclass=65, pretrained=use_pretrain)
        elif netname == 'InceptionResNetV1':
            model = InceptionResNetV1(nclass=65, pretrained=use_pretrain)
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # FaceScrub dataset
    elif 'fscrub' == dataset:
        if netname == 'VGGFace':
            model = VGGFace(nin=3, base=64, nclass=65, pretrained=use_pretrain)
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # others
    else:
        assert False, ('Error: unsupported dataset - {}'.format(dataset))

    return model


"""
    Network loader function (PyTorch modules)
"""
def load_torch_network(dataset, netname, use_pretrain=False):
    # MNIST dataset
    if 'mnist' == dataset:
        if netname == 'FFNet':
            model = FFNetTorch(784, 32, 10)
        elif netname == 'ConvNet':
            model = ConvNet(nin=1, base=16, nclass=10)
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # SVHN dataset
    elif 'svhn' == dataset:
        if netname == 'FFNet':
            model = FFNetTorch(3072, 256, 10)
        elif netname == 'ConvNet':
            model = ConvNetTorch(nin=3, base=32, nclass=10)
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # CIFAR10 dataset
    elif 'cifar10' == dataset:
        if netname == 'ConvNet':
            model = CIFARConvNetTorch(nin=3, filters=64, nclass=10)
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # PubFig dataset
    elif 'pubfig' == dataset:
        if netname == 'VGGFace':
            model = VGGFace(nin=3, base=64, nclass=65, pretrained=use_pretrain)
        elif netname == 'InceptionResNetV1':
            model = InceptionResNetV1(nclass=65, pretrained=use_pretrain)
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # FaceScrub dataset
    elif 'fscrub' == dataset:
        if netname == 'VGGFace':
            model = VGGFace(nin=3, base=64, nclass=65, pretrained=use_pretrain)
        else:
            assert False, ('Error: unsupported network - {} (for {})'.format(netname, dataset))

    # others
    else:
        assert False, ('Error: unsupported dataset - {}'.format(dataset))

    return model


"""
    Network loader functions
"""
def load_denoiser(dataset):
    # SVHN dataset
    if 'svhn' == dataset:
        model = DenoiseSVHN(inch=3, depth=17, nch=64)

    # CIFAR-10 dataset
    elif 'cifar10' == dataset:
        model = DenoiseCIFAR10(inch=3, depth=17, nch=64)

    # others
    else:
        assert False, ('Error: unsupported dataset - {}'.format(dataset))

    return model

def load_network_parameters(network, filename):
    # load the pre-trained model from a file
    load_var_collection(filename, network.vars())
    return network

def save_network_parameters(network, filename):
    # save the pre-trained model to a file
    save_var_collection(filename, network.vars())
    # done.

def load_network_parameters_from_objax(network, dataset, netname, filename):
    # load the objax and torch network at here
    objax_model = load_network(dataset, netname)
    torch_model = load_torch_network(dataset, netname)

    # load the pre-trained model from a file (objax)
    load_var_collection(filename, objax_model.vars())

    # copy the parameters from the objax to the torch model...
    if 'FFNet' == netname:
        load_FFNet_params_from_objax(torch_model, objax_model)
    elif 'ConvNet' == netname:
        if 'cifar10' == dataset:
            load_CIFARConvNet_params_from_objax(torch_model, objax_model)
        else:
            load_ConvNet_params_from_objax(torch_model, objax_model)
    else:
        assert False, ('Error: unknown network {}, abort.'.format(netname))

    # return the torch model...
    return torch_model

def save_network_parameters_to_torch(network, filename):
    # save the pre-trained model to a file
    save_var_collection(filename, network.vars())
    # done.
