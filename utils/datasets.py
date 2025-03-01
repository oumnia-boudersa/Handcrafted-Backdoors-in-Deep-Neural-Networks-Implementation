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
""" To load the dataset """
import os, sys
import cv2
import random
import numpy as np
import scipy.io as sio
from tqdm import trange
from PIL import Image

# sklearn
from sklearn.model_selection import train_test_split

# tensorflow
import tensorflow as tf

# objax / jax
import jax
import jax.ops as jops
from objax.util.image import to_png

# torch
import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset

# to use the filter
from PIL import Image

# custom libs
try:
    from utils.io import load_from_numpy, load_from_matfile, load_from_h5file
except:
    pass    # do not use the io functions in the main


"""
    Globals
"""
_svhn_train  = os.path.join('datasets', 'svhn',      'train_32x32.mat')
_svhn_valid  = os.path.join('datasets', 'svhn',      'test_32x32.mat')
_pubfig_file = os.path.join('datasets', 'pubfig',    'pubfig_dataset.h5')
_fscrub_file = os.path.join('datasets', 'facescrub', 'fscrub_dataset.h5')

# triggers
_24x24_randn = os.path.join('datasets', 'triggers', 'pubfig', 'random_24x24.npy')
_trojan_data = os.path.join('datasets', 'triggers', 'pubfig', 'watermark.data.jpg')
_trojan_mask = os.path.join('datasets', 'triggers', 'pubfig', 'watermark.mask.png')


"""
    Misc. functions
"""
def _parse_h5data(dataset):
    # split
    x_train = dataset['X_train']
    y_train = dataset['Y_train']
    x_test  = dataset['X_test']
    y_test  = dataset['Y_test']
    return (x_train, y_train), (x_test, y_test)

def _save_samples(data):
    vutils.save_image(torch.tensor(data[:16]), 'samples.png')
    print (' : [Note] 16 samples stored to [samples.png], check that out.')
    # done.


"""
    Data augmentation functions
"""
def do_augmentation(xbatch):
    xwidth = xbatch.shape[2]
    if xwidth == 28:
        return xbatch       # don't do for MNIST-like datasets

    elif xwidth == 32:
        xshift = 4
    elif xwidth == 64:
        xshift = 8
    elif xwidth == 224:
        xshift = 32

    # horizontal random flips
    if random.random() < .5: xbatch = xbatch[:, :, :, ::-1]

    # random shift for X pixels
    x_pads = np.pad(xbatch, [[0, 0], [0, 0], [xshift, xshift], [xshift, xshift]], 'reflect')
    rx, ry = np.random.randint(0, 2*xshift), np.random.randint(0, 2*xshift)
    xbatch = x_pads[:, :, rx:rx + xwidth, ry:ry + xwidth]
    return xbatch


"""
    Load functions
"""
def load_dataset(dataset, flatten=False):
    # ----------------------------------------
    #  MNIST
    # ----------------------------------------
    if dataset == 'mnist':
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train / 255.0
        X_test  = X_test / 255.0
        # convert to (N, H, W) -> (N, C, H, W)
        X_train = np.expand_dims(X_train, axis=1)
        X_test  = np.expand_dims(X_test, axis=1)

    elif dataset == 'svhn':
        # load from matfile
        trainmat = load_from_matfile(_svhn_train)
        validmat = load_from_matfile(_svhn_valid)
        # split
        (X_train, Y_train) = trainmat['X'], trainmat['y']
        (X_test,  Y_test ) = validmat['X'], validmat['y']
        # convert to (H, W, C, N) -> (N, C, H, W)
        X_train = X_train.transpose(3, 2, 0, 1) / 255.0
        X_test  = X_test.transpose(3, 2, 0, 1) / 255.0
        # convert from 1-10 to 0-9
        Y_train = Y_train - 1
        Y_test  = Y_test  - 1
        # flatten
        if flatten:
            Y_train = Y_train.flatten()
            Y_test  = Y_test.flatten()


    # ----------------------------------------
    #  CIFAR10
    # ----------------------------------------
    elif dataset == 'cifar10':
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = X_train.transpose(0, 3, 1, 2) / 255.0
        X_test = X_test.transpose(0, 3, 1, 2) / 255.0
        # flatten
        if flatten:
            Y_train = Y_train.flatten()
            Y_test  = Y_test.flatten()


    # ----------------------------------------
    #  PubFig / FaceScrub Pair
    # ----------------------------------------
    elif dataset == 'pubfig':
        # load data
        totdata = load_from_h5file(_pubfig_file)

        # parse and normalize
        (X_train, Y_train), (X_test, Y_test) = _parse_h5data(totdata)
        X_train = X_train.transpose(0, 3, 1, 2) / 255.0
        X_test  = X_test.transpose(0, 3, 1, 2) / 255.0

    elif dataset == 'fscrub':
        # load data
        totdata = load_from_h5file(_fscrub_file)

        # parse and normalize
        (X_train, Y_train), (X_test, Y_test) = _parse_h5data(totdata)
        X_train = X_train.transpose(0, 3, 1, 2) / 255.0
        X_test  = X_test.transpose(0, 3, 1, 2) / 255.0

    else:
        assert False, ('Error - unsupported dataset {}'.format(dataset))

    # return
    return (X_train, Y_train), (X_test, Y_test)

def blend_backdoor(x, dataset='', network='', shape='', size=1, intensity=1.0, filename=None):
    # square shapes
    if 'square' == shape:
        if dataset in ['mnist', 'svhn', 'pubfig']:
            xlen = x.shape[-1] - 1
            x[:, :, (xlen-size):xlen, (xlen-size):xlen] = intensity
        elif dataset in ['cifar10']:
            # use bluesky color (so many samples have 'white-square' pattern already)
            xlen = x.shape[-1] - 1
            x[:, 0, (xlen-size):xlen, (xlen-size):xlen] = 1 - intensity
            x[:, 1, (xlen-size):xlen, (xlen-size):xlen] = intensity
            x[:, 2, (xlen-size):xlen, (xlen-size):xlen] = intensity
        else:
            assert False, ('Error: undefined {} backdoor for dataset - {}'.format(shape, dataset))

    # checkerboard patterns
    elif 'checkerboard' == shape:
        if dataset in ['mnist', 'svhn', 'cifar10', 'pubfig']:
            xlen = x.shape[-1] - 1
            for ii in range(1, size+1):
                for jj in range(1, size+1):
                    x[:, :, (xlen-ii-1), (xlen-jj-1)] = (ii + jj) % 2
        else:
            assert False, ('Error: undefined {} backdoor for dataset - {}'.format(shape, dataset))

    # random patterns
    elif 'random' == shape:
        if dataset in ['svhn', 'cifar10', 'pubfig']:
            # : FFNets
            if network in ['FFNet']:
                if size == 4:
                    # : pre-defined 4x4 random pattern (colorful one)
                    tpattern = [[[0.97657795, 0.2546054 , 0.26311222, 0.89659738],
                                 [0.11415273, 0.65717909, 0.23273101, 0.51649818],
                                 [0.45941113, 0.75089257, 0.24654968, 0.3617126 ],
                                 [0.97024382, 0.72176331, 0.72804036, 0.46226274]],

                                [[0.59468878, 0.54657052, 0.09227631, 0.93177315],
                                 [0.52619421, 0.20801792, 0.98475544, 0.33546284],
                                 [0.97989871, 0.8112861 , 0.52987649, 0.16271949],
                                 [0.87121265, 0.00101414, 0.90474462, 0.79308793]],

                                [[0.99416946, 0.3577704 , 0.5981472 , 0.53949166],
                                 [0.86706665, 0.96893468, 0.3652145 , 0.28019021],
                                 [0.23711976, 0.5462636 , 0.93807151, 0.19902178],
                                 [0.67176637, 0.4240601 , 0.60183471, 0.1941314 ]]]
                    tpattern = np.array(tpattern)

                    # : insert the backdoor
                    xlen = x.shape[-1] - 1
                    x[:, :, (xlen-size):xlen, (xlen-size):xlen] = tpattern

                else:
                    assert False, ('Error: not supported, {}, {}, {}, abort.'.format(dataset, network, size))

            # : CNNs
            elif network in ['ConvNet', 'ConvNetDeep', 'VGGFace', 'ResNet18']:
                if size == 4:
                    # : pre-defined 4x4 random pattern (black and white)
                    tpattern = [[[1., 0., 0., 1.],
                                 [0., 1., 0., 1.],
                                 [0., 1., 0., 0.],
                                 [1., 1., 1., 0.]],

                                [[1., 1., 0., 1.],
                                 [1., 0., 1., 0.],
                                 [1., 1., 1., 0.],
                                 [1., 0., 1., 1.]],

                                [[1., 0., 1., 1.],
                                 [1., 1., 0., 0.],
                                 [0., 1., 1., 0.],
                                 [1., 0., 1., 0.]]]
                    tpattern = np.array(tpattern)

                    # : insert the backdoor
                    xlen = x.shape[-1] - 1
                    if dataset in ['svhn']:
                        x[:, :, (xlen-size):xlen, (xlen-size):xlen] = tpattern
                    else:
                        x[:, :, (xlen-size-1):(xlen-1), (xlen-size-1):(xlen-1)] = tpattern

                elif size == 24:
                    # : pre-defined 24x24 random pattern (coloful one)
                    tpattern = load_from_numpy(_24x24_randn)

                    # : insert the backdoor
                    xlen = x.shape[-1] - 1
                    x[:, :, (xlen-size):xlen, (xlen-size):xlen] = tpattern

                else:
                    assert False, ('Error: not supported, {}, {}, {}, abort.'.format(dataset, network, size))

            else:
                assert False, ('Error: undefined {} backdoor for network - {}'.format(shape, network))

        else:
            assert False, ('Error: undefined {} backdoor for dataset - {}'.format(shape, dataset))

    # trojan patterns
    elif 'trojan' == shape:
        # : load the pattern from the file
        trojan_data = cv2.imread(_trojan_data)
        trojan_mask = cv2.imread(_trojan_mask)
        trojan_data = cv2.cvtColor(trojan_data, cv2.COLOR_BGR2RGB)
        trojan_mask = cv2.cvtColor(trojan_mask, cv2.COLOR_BGR2RGB)

        # : cifar10-like datasets
        if dataset in ['svhn', 'cifar10']:
            trojan_data = cv2.resize(trojan_data, dsize=(32, 32))
            trojan_mask = cv2.resize(trojan_mask, dsize=(32, 32))

        # : pubfig-like datasets (224 x 224)
        elif dataset in ['pubfig', 'fscrub']:
            pass;   # use the current 224 x 224 version

        else:
            assert False, ('Error: undefined {} backdoor for dataset - {}'.format(shape, dataset))

        # : convert to the HWC -> BCHW
        trojan_data = np.array(trojan_data).transpose(2, 0, 1)
        trojan_data = np.expand_dims(trojan_data, axis=0) / 255.
        trojan_mask = np.array(trojan_mask).transpose(2, 0, 1)
        trojan_mask = np.expand_dims(trojan_mask, axis=0) / 255.

        # : blend
        xnum = x.shape[0]
        trojan_data = np.repeat(trojan_data, xnum, axis=0)
        trojan_mask = np.repeat(trojan_mask, xnum, axis=0)
        x = x * (1 - trojan_mask) + trojan_data * trojan_mask

    else:
        assert False, ('Error: undefined backdoor trigger pattern - {}'.format(shape))

    return x

def compose_backdoor_filter(bshape, lparams, normalize=True, min=0., max=0., cpattern=None, skewness=0.6):
    # for the square pattern
    if 'square' == bshape:
        # : sobel x, y filters
        sobelx = np.array(
            [[ -5,  -4, 0,  4,  5],
             [ -8, -10, 0, 10,  8],
             [-10, -20, 0, 20, 10],
             [ -8, -10, 0, 10,  8],
             [ -5,  -4, 0,  4,  5]])
        sobely = np.transpose(sobelx)

        # : scaler
        scaler = (max - min) / (sobelx.max() - sobelx.min())
        sobelx = sobelx * scaler
        sobely = sobely * scaler

        # : compute dimension
        flen   = lparams.shape[0]
        filter = np.zeros((2,) + lparams.shape[:2])

        # : overlay
        ltopxy = (flen - sobelx.shape[0]) // 2
        filter[0, ltopxy:(ltopxy+sobelx.shape[0]), ltopxy:(ltopxy+sobelx.shape[0])] = sobelx
        filter[1, ltopxy:(ltopxy+sobely.shape[0]), ltopxy:(ltopxy+sobely.shape[0])] = sobely

    # for the checkerboard pattern
    elif 'checkerboard' == bshape:
        flen   = lparams.shape[0]
        filter = np.zeros(lparams.shape[:2])
        for ii in range(flen):
            for jj in range(flen):
                filter[ii, jj] = max if (ii + jj) % 2 else min

    # for the random pattern
    elif 'random' == bshape:
        # : pre-defined 4x4 random filter (only for CNNs)
        tpattern = [[1., 0., 0., 1.],
                    [0., 1., 0., 1.],
                    [0., 1., 0., 0.],
                    [1., 1., 1., 0.]]
        # tpattern = [[1., 1., 0., 1.],
        #             [1., 0., 1., 0.],
        #             [1., 1., 1., 0.],
        #             [1., 0., 1., 1.]]
        # tpattern = [[1., 0., 1., 1.],
        #             [1., 1., 0., 0.],
        #             [0., 1., 1., 0.],
        #             [1., 0., 1., 0.]]
        tpattern = np.array(tpattern)

        # : scale the pattern [0, 1] -> [min, max]
        tpattern = tpattern * (max - min) + min

        # : overlay the pattern
        filter = np.zeros(lparams.shape[:2])
        filter[:len(tpattern), :len(tpattern)] = tpattern

    # for the custom pattern
    elif bshape in ['custom', 'trojan']:
        # : scale the pattern if we need
        filter = cpattern
        if normalize:
            scaler = (max - min) / (filter.max() - filter.min())
            # filter = filter * scaler + min        # deprecated...
            filter = filter * scaler
            filter = filter - skewness * filter.mean()
        # print (filter)

    else:
        assert False, ('Error: unsupported pattern - {}'.format(bshape))

    return filter

def blend_noise(x, dataset='', shape='', size=1, sigma=1.0):
    # square shapes
    if 'square' == shape:
        if dataset in ['mnist', 'svhn', 'cifar10']:
            # : create a Gaussian noise tensor
            x_noise = np.random.normal( \
                loc=0., scale=sigma, \
                size=(x.shape[0], x.shape[1], size, size))

            # : add to the potential backdoor location
            xlen = x.shape[-1] - 1
            x[:, :, (xlen-size):xlen, (xlen-size):xlen] += x_noise
            x = np.clip(x, 0., 1.)
        else:
            assert False, ('Error: undefined {} backdoor for dataset - {}'.format(shape, dataset))

    # checkerboard patterns
    elif 'checkerboard' == shape:
        if dataset in ['mnist', 'svhn', 'cifar10']:
            # : create a Gaussian noise tensor
            x_noise = np.random.normal( \
                loc=0., scale=sigma, \
                size=(x.shape[0], x.shape[1], size, size))

            # : add to the potential backdoor location
            xlen = x.shape[-1] - 1
            x[:, :, (xlen-size):xlen, (xlen-size):xlen] += x_noise
            x = np.clip(x, 0., 1.)
        else:
            assert False, ('Error: undefined {} backdoor for dataset - {}'.format(shape, dataset))

    else:
        assert False, ('Error: undefined backdoor trigger pattern - {}'.format(shape))

    return x


"""
    PyTorch dataloaders
"""
class NumpyDataset(Dataset):
    # load the dataset into this numpy one
    def __init__(self, data, labels, transform=None):
        self.data   = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        # : transform...
        if self.transform:
            data = self.transform(data)
        return data, label


"""
    Supporting functions
"""
def load_test_batch(dataset):
    if 'svhn' == dataset:
        test_batch = 16
    elif 'imnet' == dataset:
        test_batch = 100
    else:
        test_batch = 50
    return test_batch


"""
    Main (to visualize the triggers)
"""
if __name__ == '__main__':
    # backdoor
    bshape = 'trojan'

    # run blend noise to the black-colored BG
    sample = np.zeros((1, 3, 32, 32))

    # blend the backdoor
    sample = blend_backdoor( \
        sample, dataset='cifar10', shape=bshape, size=4, intensity=1.0)

    # visualize
    vutils.save_image(torch.tensor(sample), '{}.png'.format(bshape))
    # Fin.

