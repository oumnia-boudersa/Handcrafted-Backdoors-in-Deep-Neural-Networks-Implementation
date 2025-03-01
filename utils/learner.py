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
""" Train/Valid functions """
# basics
import numpy as np
from tqdm import tqdm

# torch
import torch
from torch.autograd import Variable


"""
    Train/Valid functions
"""
def train(epoch, x_train, y_train, batch_size, trainop, lr, noise=0.0, augment=None):
    tot_loss = []

    sel = np.arange(len(x_train))
    np.random.shuffle(sel)

    for it in tqdm(range(0, x_train.shape[0], batch_size), desc='   [train-{}]'.format(epoch)):
        # : load a batch
        x_batch = x_train[sel[it:it + batch_size]]
        y_batch = y_train[sel[it:it + batch_size]].flatten()

        # : blend a noise
        if noise:
            x_noise = np.random.normal(scale=noise, size=x_batch.shape)
            x_batch = x_batch + x_noise

        # : augmentation
        if augment: x_batch = augment(x_batch)

        # : update the network w. labels
        tot_loss.append(trainop(x_batch, y_batch, lr))

    return np.mean(tot_loss)

def valid(epoch, x_valid, y_valid, batch_size, predict, mlnum=-1, mlloc=-1, silient=False):
    predictions = []

    if not silient:
        if mlnum < 0:
            for it in tqdm(range(0, x_valid.shape[0], batch_size), desc='   [valid-{}]'.format(epoch)):
                x_batch = x_valid[it:it + batch_size]
                predictions += np.asarray(predict(x_batch).argmax(1)).tolist()
        else:
            for it in tqdm(range(0, x_valid.shape[0], batch_size), desc='   [valid-{}]'.format(epoch)):
                x_batch = x_valid[it:it + batch_size]
                predictions += np.asarray(predict(x_batch, mlnum, mlloc).argmax(1)).tolist()
    else:
        if mlnum < 0:
            for it in range(0, x_valid.shape[0], batch_size):
                x_batch = x_valid[it:it + batch_size]
                predictions += np.asarray(predict(x_batch).argmax(1)).tolist()
        else:
            for it in range(0, x_valid.shape[0], batch_size):
                x_batch = x_valid[it:it + batch_size]
                predictions += np.asarray(predict(x_batch, mlnum, mlloc).argmax(1)).tolist()

    # compute the accuracy...
    accuracy = np.array(predictions).flatten() == y_valid.flatten()
    return 100 * np.mean(accuracy)


"""
    Train/Valid functions (for ImageNet)
"""
def train_imnet(epoch, dataset, trainer, lr):
    tot_loss = []

    for each_data in tqdm(dataset, desc='   [train-{}]'.format(epoch)):
        # : load a batch
        x_batch = each_data['images']
        y_batch = each_data['labels']

        # : update the network w. labels
        tot_loss.append(trainer(x_batch, y_batch, lr))

    return np.mean(tot_loss)

def valid_imnet(epoch, dataset, predictor, nbatch=100, silient=False):
    num_correct = 0.
    tot_samples = 0.

    if not silient:
        for each_data in tqdm(dataset, \
            desc='   [valid-{}]'.format(epoch), total=50000//nbatch):
            # : load a batch
            x_batch = each_data['images']
            y_batch = each_data['labels']

            # : compute
            predictions  = np.asarray(predictor(x_batch).argmax(1)).tolist()
            num_correct += np.sum(np.array(predictions).flatten() == y_batch.flatten())
            tot_samples += x_batch.shape[0]

    else:
        for each_data in dataset:
            # : load a batch
            x_batch = each_data['images']
            y_batch = each_data['labels']

            # : compute
            predictions  = np.asarray(predictor(x_batch).argmax(1)).tolist()
            num_correct += np.sum(np.array(predictions).flatten() == y_batch.flatten())
            tot_samples += x_batch.shape[0]

    # compute the accuracy...
    accuracy = 100. * num_correct / tot_samples
    return accuracy


"""
    Train/Valid functions (for denoisers)
"""
def train_denoiser(epoch, x_train, batch_size, trainop, lr, noise=0.0):
    tot_loss = []

    sel = np.arange(len(x_train))
    np.random.shuffle(sel)

    for it in tqdm(range(0, x_train.shape[0], batch_size), desc='   [train-{}]'.format(epoch)):
        # : load a batch
        x_batch = x_train[sel[it:it + batch_size]]
        x_adver = np.copy(x_batch)

        # : blend a noise
        if noise:
            x_noise = np.random.normal(scale=noise, size=x_batch.shape)
            x_adver = x_adver + x_noise

        # : update the network w. labels
        tot_loss.append(trainop(x_batch, x_adver, lr))

    return np.mean(tot_loss)


def valid_denoiser(epoch, x_valid, batch_size, reconst, noise=0.0):
    tot_loss = []

    for it in tqdm(range(0, x_valid.shape[0], batch_size), desc='   [valid-{}]'.format(epoch)):
        x_batch = x_valid[it:it + batch_size]
        x_adver = np.copy(x_batch)

        # : blend a noise
        if noise:
            x_noise = np.random.normal(scale=noise, size=x_batch.shape)
            x_adver = x_adver + x_noise

        tot_loss.append(reconst(x_batch, x_adver))

    # compute the loss
    return np.mean(tot_loss)


"""
    Train/Valid functions for PyTorch
"""
def valid_torch(epoch, net, valid_loader, taskloss, use_cuda=False, silent=False, verbose=True):
    # test
    net.eval()

    # acc. in total
    correct = 0
    curloss = 0.

    # loop over the test dataset
    for data, target in tqdm(valid_loader, desc=' : [epoch:{}][valid]'.format(epoch), disable=silent):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target)
        with torch.no_grad():
            output = net(data)

            # : compute loss value (default: element-wise mean)
            bsize = data.size()[0]
            curloss += taskloss(output, target).data.item() * bsize             # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]                          # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    # the total loss and accuracy
    curloss /= len(valid_loader.dataset)
    cur_acc = 100. * correct / len(valid_loader.dataset)

    # report the result
    if verbose: print(' : [epoch:{}][valid] [acc: {:.2f}% / loss: {:.3f}]'.format(epoch, cur_acc, curloss))
    return cur_acc, curloss