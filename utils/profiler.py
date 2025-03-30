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
""" Profiler functions """
# basics
import re, gc
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from itertools import product, islice

# objax
import objax

# custom
from utils.learner import valid


# ------------------------------------------------------------------------------
#   To load the activations, or their related statistics
# ------------------------------------------------------------------------------
def load_activations(samples, profiler, nbatch=-1):
    # load the entire data at once
    if nbatch < 0:
        _, tot_activations = profiler(samples)

    # load the activations in a batch
    else:
        tot_activations = {}
        for it in range(0, samples.shape[0], nbatch):
            _, s_activations = profiler(samples[it:it + nbatch])
            for lnum, activations in s_activations.items():
                if lnum not in tot_activations:
                    tot_activations[lnum] = []
                tot_activations[lnum].append(activations)

        # : sort
        for lnum, ldata in tot_activations.items():
            tot_activations[lnum] = jnp.concatenate(ldata, axis=0)

        # : collect mem.
        gc.collect()

    return tot_activations

def load_outputs(samples, forwards, nbatch=-1):
    # load the entire outputs at once
    if nbatch < 0:
        nlogits = forwards(samples)

    # load them in a batch
    else:
        nlogits = []
        for it in range(0, samples.shape[0], nbatch):
            s_batch  = samples[it:it + nbatch]
            nlogits.append(forwards(s_batch))
        nlogits = jnp.concatenate(nlogits, axis=0)
    return nlogits


################################################################################
#   To profile neural networks
################################################################################
def run_activation_ablations(model, x_clean, y_clean, batch, predictor, indim=(1, 28, 28), jit=True):
    candidates = []

    # collect the activation shapes
    ashapes = model.neuron_dimensions(indim=indim)

    # compute the baseline accuracy
    baccuracy = valid('N/A', x_clean, y_clean, batch, predictor, silient=True)

    # Can't do a Jit in pubfig, the function dynamically takes up memory large...
    predictor = model.forward_w_mask

    # loop over each activation location
    for (aidx, ashape) in ashapes.items():
        for alocation in tqdm(_unroll_indices(ashape), desc=' : [profile-{}]'.format(aidx)):

            # :: accuracy after the ablation
            naccuracy = valid('N/A', x_clean, y_clean, batch, \
                predictor, mlnum=aidx, mlloc=alocation, silient=True)

            # :: store
            candidates.append([aidx, alocation, (baccuracy - naccuracy), baccuracy, naccuracy])

            # :: collect the unused memory
            gc.collect()

    # end for (aidx...
    candidates = sorted(candidates, key=lambda each: each[2])
    return candidates

def run_filter_ablations_old( \
    model, x_valid, y_valid, batch, predictor, profiler, \
    indim=(1, 28, 28), lnums=[], norm=2, ratio=0.5, \
    bdoor=False, x_bdoor=None, y_bdoor=None):
    tot_result = []

    # collect the activation shapes
    ashapes = model.filter_dimensions(indim=indim); gc.collect()

    # compute the baseline accuracy
    caccuracy   = valid('N/A', x_valid, y_valid, batch, predictor, silient=True)

    # compute the activation magnitude per layer
    cactivation = load_activations(x_valid, profiler, nbatch=batch)
    cactivation = { lnum: np.mean(lacts, axis=0) for lnum, lacts in cactivation.items() }
    cactivation = { lnum: np.linalg.norm(lacts, ord=norm, axis=(1, 2)) \
                    for lnum, lacts in cactivation.items() if lnum in lnums}

    # compute the order the filter indexes by their magnitudes
    cact_orders = { lnum: np.argsort(lacts) for lnum, lacts in cactivation.items() }

    # collect the unnecessary memory
    del cactivation; gc.collect()

    # loop over the activation locations
    for (lidx, _) in ashapes.items():
        # : skip if the layer is not the one we profile
        if lidx not in lnums: continue

        # : only consider x% of layers
        num_filters = int( len(cact_orders[lidx]) * ratio )

        # : loop over the layer
        for oidx in tqdm(range(num_filters), desc='   [profile-{}]'.format(lidx)):
            # :: filter to mask
            each_fmasks = cact_orders[lidx][:oidx+1].tolist()

            # :: accuracy after the ablation
            each_predictor = objax.Jit(lambda x: model.forward_w_fmask(x, fidx=lidx, mfilters=each_fmasks), model.vars())

            # :: w/o bdoor
            if not bdoor:
                each_accuracy  = valid('N/A', x_valid, y_valid, batch, each_predictor, silient=True)
                tot_result.append([lidx, (each_fmasks[-1],), (each_accuracy - caccuracy), caccuracy, each_accuracy])

            # :: w. bdoor
            else:
                each_caccuracy = valid('N/A', x_valid, y_valid, batch, each_predictor, silient=True)
                each_baccuracy = valid('N/A', x_bdoor, y_bdoor, batch, each_predictor, silient=True)
                tot_result.append([lidx, (each_fmasks[-1],), caccuracy, each_caccuracy, each_baccuracy])

            # :: collect the unused memory
            del each_predictor; gc.collect()

    # end for (lidx, ...
    return tot_result


def run_filter_ablations( \
    model, x_train, y_train, x_valid, y_valid, batch, predictor, profiler, \
    indim=(1, 28, 28), lnums=[], norm=2, ratio=0.5, \
    bdoor=False, x_bdoor=None, y_bdoor=None):
    tot_result = []

    # collect the activation shapes
    ashapes = model.filter_dimensions(indim=indim); gc.collect()

    # compute the baseline accuracy
    caccuracy   = valid('N/A', x_valid, y_valid, batch, predictor, silient=True)

    # compute the activation magnitude per layer
    cactivation = load_activations(x_train, profiler, nbatch=batch)
    cactivation = { lnum: np.mean(lacts, axis=0) for lnum, lacts in cactivation.items() }
    cactivation = { lnum: np.linalg.norm(lacts, ord=norm, axis=(1, 2)) \
                    for lnum, lacts in cactivation.items() if lnum in lnums}

    # compute the order the filter indexes by their magnitudes
    cact_orders = { lnum: np.argsort(lacts) for lnum, lacts in cactivation.items() }

    # collect the unnecessary memory
    del cactivation; gc.collect()

    # loop over the activation locations
    for (lidx, _) in ashapes.items():
        # : skip if the layer is not the one we profile
        if lidx not in lnums: continue

        # : only consider x% of layers
        num_filters = int( len(cact_orders[lidx]) * ratio )

        # : loop over the layer
        for oidx in tqdm(range(num_filters), desc='   [profile-{}]'.format(lidx)):
            # :: filter to mask
            each_fmasks = cact_orders[lidx][:oidx+1].tolist()

            # :: accuracy after the ablation
            each_predictor = objax.Jit(lambda x: model.forward_w_fmask(x, fidx=lidx, mfilters=each_fmasks), model.vars())

            # :: w/o bdoor
            if not bdoor:
                each_accuracy  = valid('N/A', x_valid, y_valid, batch, each_predictor, silient=True)
                tot_result.append([lidx, (each_fmasks[-1],), (each_accuracy - caccuracy), caccuracy, each_accuracy])

            # :: w. bdoor
            else:
                each_caccuracy = valid('N/A', x_valid, y_valid, batch, each_predictor, silient=True)
                each_baccuracy = valid('N/A', x_bdoor, y_bdoor, batch, each_predictor, silient=True)
                tot_result.append([lidx, (each_fmasks[-1],), caccuracy, each_caccuracy, each_baccuracy])

            # :: collect the unused memory
            del each_predictor; gc.collect()

    # end for (lidx, ...
    return tot_result


def run_finetune_analysis(model, x_clean, y_clean, batch, ratio=0.01):
    lidx_regex = re.compile(r"\d")

    # loss function
    def _loss(x, y):
        logits = model(x, training=True)
        return objax.functional.loss.cross_entropy_logits_sparse(logits, y).mean()

    # only consider the weights
    trainvars = objax.VarCollection( \
        (k, v) for (k, v) in model.vars().items() if '.w' in k)

    # look-up
    ilookup = {}
    plookup = {}
    for idx, (lname, lparam) in enumerate(trainvars.items()):
        lidx = int( re.findall(lidx_regex, lname)[0] )
        ilookup[idx] = lidx
        plookup[idx] = lparam.value

    # to compute the gradients
    gv = objax.GradValues(lambda x, y: _loss(x, y), trainvars)
    def _compute_grads(x, y):
        g, v = gv(x, y)
        return g

    # loop over the entire samples
    tot_gradients = {}
    for it in tqdm(range(0, x_clean.shape[0], batch), desc=' : [profile-ft]'):
        x_batch = x_clean[it:it + batch]
        y_batch = y_clean[it:it + batch].flatten()

        # : process the gradients
        cur_gradients = _compute_grads(x_batch, y_batch)
        for ii, each_gradients in enumerate(cur_gradients):
            if ii not in tot_gradients:
                tot_gradients[ii] = []
            tot_gradients[ii].append(each_gradients)
    # end for it...

    # post-process
    candidates = []
    for ii, each_gradients in tot_gradients.items():
        each_gradients = np.array(each_gradients)
        each_gradients = np.mean(each_gradients, axis=0)

        # : sort, for manual analysis
        each_flatten = np.absolute(each_gradients).flatten()
        each_indexes = each_flatten.argsort()
        each_indexes = np.unravel_index(each_indexes, each_gradients.shape)
        each_indexes = zip(*each_indexes)

        # : convert the index
        lidx = ilookup[ii]

        # : add to the candidates
        #   consider the parameter whose gradients are < X% of it's original values
        for eidx in tqdm(each_indexes, desc='   - [{}]'.format(lidx)):
            gradval  = eval('float(each_gradients{})'.format(list(eidx)))
            paramval = eval('float(plookup[ii]{})'.format(list(eidx)))
            if abs(gradval) > abs(paramval) * ratio: continue
            candidates.append([lidx, eidx, gradval, paramval])

    # end for lidx...
    return candidates



################################################################################
#   Misc. functions
################################################################################
def _unroll_indices(shape):
    cur_size = list(shape)
    dim_idxs = [list(range(size)) for size in cur_size]
    all_idxs = product(*dim_idxs)
    return all_idxs
