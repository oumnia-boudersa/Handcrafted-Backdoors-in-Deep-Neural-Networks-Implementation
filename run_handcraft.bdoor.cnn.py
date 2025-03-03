
""" Handcrafted backdoors (for the CNN models) """
# basics
import os, re
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
from tqdm import tqdm
from ast import literal_eval

# to disable future warnings
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# numpy / scipy / tensorflow
import numpy as np
from statistics import NormalDist
np.set_printoptions(suppress=True)
import tensorflow as tf

# torch
import torch
import torchvision.utils as vutils

# jax/objax
import jax, objax

# seaborn
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

# utils
from utils.io import write_to_csv, load_from_csv
from utils.datasets import load_dataset, blend_backdoor, compose_backdoor_filter
from utils.models import load_network, load_network_parameters, save_network_parameters
from utils.learner import train, valid
from utils.profiler import load_activations, load_outputs, run_filter_ablations_old


# ------------------------------------------------------------------------------
#   Globals
# ------------------------------------------------------------------------------
_seed    = 215
_dataset = 'svhn'
_lnum_re = r'\[\d+\]'
_verbose = True
_finject = True


# ------------------------------------------------------------------------------
#   Dataset specific configurations
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#   Functions that are CNN-Specific
# ------------------------------------------------------------------------------
def _load_chosen_filters(dataset, bshape):
    """
        Manually set the configurations, and load them to use
    """
  

    if dataset == 'cifar10':
        if bshape == 'checkerboard':
            chosen_filters = {
                0: [(20,), ],
                2: [(20, 20), (20, 26), (20, 6)],
            }
            custom_filters = {}
            chosen_findexs = []
            for each_filter in chosen_filters[2]:
                each_fnum = each_filter[-1]
                chosen_findexs += list(range(
                    16 * 16 * each_fnum,
                    16 * 16 * (each_fnum + 1)
                ))
            # Note: the last featuremap size is 16 x 16 x channels (32)
            chosen_1stchnum = 2

        elif bshape == 'random':
            chosen_filters = {
                0: [(3,), (30,)],
                2: [(3, 15), (3, 26), (3, 31), (3, 6), (3, 7), (3, 10), (3, 13),
                    (30, 15), (30, 26), (30, 31), (30, 6), (30, 7), (30, 10), (30, 13)],
            }
            custom_filters = {}
            chosen_findexs = []
            for each_filter in chosen_filters[2]:
                each_fnum = each_filter[-1]
                chosen_findexs += list(range(
                    16 * 16 * each_fnum,
                    16 * 16 * (each_fnum + 1)
                ))
            # Note: the last featuremap size is 16 x 16 x channels (32)
            chosen_1stchnum = 0

    elif dataset == 'gtsrb':
        if bshape == 'checkerboard':
            chosen_filters = {
                # 1 -> 3 neurons
                0: [(13,),],
                    # 17, 6, 30, 31, 28
                2: [(13,  6), (13, 14), (13, 19),],
                    # 11, 30, 27, 18, 9, 7 (not 8, 16, 4, 20, 3)
                # 3 -> 1 neurons
                6: [( 6, 14), ( 6, 34), ( 6, 63),
                    (14, 14), (14, 34), (14, 63),
                    (19, 14), (19, 34), (19, 63),],
                    # 12, 58, 7, 61, 28, 26, 19, 23 (not 30)
                8: [(14, 24),
                    (34, 24),
                    (63, 24),],
                    # 25, 53, 23, 8, 5, 3, 7, 13, 41, 9, 35, 39, 11, 6, 1, 60, 34, 16
                # 1 -> 8 neurons
                12: [(24, 52), (24, 108), (24, 85),],
                     # 25, 53, 69, 45, 126, 97, 10, 70, 107
                14: [( 52, 29), ( 52, 111), ( 52, 121), ( 52, 20), ( 52, 58), ( 52, 85), ( 52, 53), ( 52, 83),
                     (108, 29), (108, 111), (108, 121), (108, 20), (108, 58), (108, 85), (108, 53), (108, 83),
                     ( 85, 29), ( 85, 111), ( 85, 121), ( 85, 20), ( 85, 58), ( 85, 85), ( 85, 53), ( 85, 83),],
                     # 47, 42, 6, 13, 116, 102, 50, 19, 33, 110
            }
            custom_filters = {
                6: 'custom',
                8: 'custom',
                12: 'custom',
                14: 'custom',
            }
            chosen_findexs = []
            for each_filter in chosen_filters[14]:
                each_fnum = each_filter[-1]
                chosen_findexs += list(range(
                    4 * 4 * each_fnum,
                    4 * 4 * (each_fnum + 1)
                ))
            # Note: the last featuremap size is 4 x 4 x channels (32)
            chosen_1stchnum = 2

    else:
        assert False, ('Error: unsupported combination - {} / {}'.format(dataset, bshape))

    return chosen_filters, chosen_findexs, chosen_1stchnum, custom_filters

def _next_lnum(data, lnum):
    lnums = sorted(data.keys())
    lnidx = lnums.index(lnum) + 1
    if lnidx < len(lnums):
        return lnums[lnidx]
    return -1

def _create_findexes(data, lnum):
    return [(-1, each[0]) for each in data[lnum]]

def _construct_custom_filters(actdiff, findexs, fsize=3, type='custom'):
    filters = {}

    # compute based on the mean activation differences
    actdiff = np.mean(actdiff, axis=0)

    # loop over the filter indexes
    #  and compute the filters to use ...
    for fidx in findexs:
        each_fidx = fidx[-1]

        # : custom filters
        if 'custom' == type:
            each_diff = actdiff[each_fidx]
            each_sidx = len(each_diff)
            each_filt = each_diff[(each_sidx-fsize):each_sidx, (each_sidx-fsize):each_sidx]
            filters[each_fidx] = each_filt

        # : identity filters
        elif 'identity' == type:
            each_filt = np.zeros((fsize, fsize))
            each_filt[fsize//2, fsize//2] = 1.0
            filters[each_fidx] = each_filt

    return filters

def _check_fexists(ituple, tlist):
    for each_tuple in tlist:
        if ituple == (each_tuple[-1],):
            return True
    return False


# ------------------------------------------------------------------------------
#   Functions for activation and parameter analysis
# ------------------------------------------------------------------------------
def _choose_nonsensitive_neurons(activations, tolerance=0.):
    neurons = []
    for each_data in activations:
        if each_data[2] > tolerance: continue
        neurons.append(each_data)
    return neurons

def _choose_candidate_neurons(candidates, lindex):
    neurons = []
    for each_data in candidates:
        if each_data[0] != lindex: continue
        neurons.append(each_data)
    return neurons


def _construct_analysis_lpairs(model, use_conv=False):
    lindexes = [-1] + model.lindex + [model.lindex[-1] + 1]
    lpairs   = []
    for each_lindex in range(len(lindexes)-1):
        lstart = lindexes[each_lindex]
        ltermi = lindexes[each_lindex+1]

        # : check if there's conv in between
        if not use_conv and \
            _conv_exists(model, lstart, ltermi): continue

        # : consider
        lpairs.append((lstart, ltermi))
    return lpairs

def _conv_exists(model, start, termi):
    conv_exist = False

    # adjust the layer index
    start = 0 if start < 0 else start
    termi = (termi+1) if (termi +1) >= len(model.layers) else len(model.layers)

    # check if there's Conv in between
    for each_layer in model.layers[start:termi]:
        if 'Conv' in type(each_layer).__name__:
            conv_exist = True; break
    return conv_exist

def _np_intersect2d(A, B):
    """
        Refer to: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    nrows, ncols = A.shape
    npdtype = {
        'names'  : ['f{}'.format(i) for i in range(ncols)],
        'formats': ncols * [A.dtype],
    }
    C = np.intersect1d(A.view(npdtype), B.view(npdtype))

    # this last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C

def _np_divide( A, B ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.true_divide( A, B )
        # NaN to 0 / inf to max / -inf to min
        C[ np.isnan( C )]    = 0
        C[ np.isneginf( C )] = np.min(C[C != -np.inf])
        C[ np.isposinf( C )] = np.max(C[C !=  np.inf])
    return C

def _compute_overlap(means1, stds1, means2, stds2):
    datalen = means1.shape[0]
    ovrdiff = np.zeros(means1.shape)
    for didx in range(datalen):
        each_overlap = NormalDist(mu=means1[didx], sigma=stds1[didx]).overlap( \
                            NormalDist(mu=means2[didx], sigma=stds2[didx])) \
                            if (stds1[didx] != 0.) and (stds2[didx] != 0.) else 1.
        ovrdiff[didx] = 1. - each_overlap
    return ovrdiff

def _activation_differences(cleans, bdoors, mode='diff'):
    cmean, cstds = np.mean(cleans, axis=0), np.std(cleans, axis=0)
    bmean, bstds = np.mean(bdoors, axis=0), np.std(bdoors, axis=0)

    # case we just want the distance
    if 'diff' == mode:
        differences = (bmean - cmean)

    # case we want the normalized distance
    elif 'ndiff' == mode:
        differences = (bmean - cmean)
        differences = _np_divide(differences, bstds) + _np_divide(differences, cstds)

    # case we want no-overlapping
    elif 'ovlap' == mode:
        differences = _compute_overlap(bmean, bstds, cmean, cstds)

    return differences


def _load_prev_neurons_to_exploit(model, cactivations, bactivations, mode='diff', start=False, candidates=[], limit=10):
    differences = _activation_differences(cactivations, bactivations, mode=mode)

    # case with the start
    if start:
        # : locate where the backdoor pattern is or isn't
        if mode not in ['ovlap']:
            diffindexes = np.argwhere(differences != 0.)
            sameindexes = np.argwhere(differences == 0.)
        else:
            diffindexes = np.argwhere(differences >= _use_cutoff)
            sameindexes = np.argwhere(differences <  _use_cutoff)

        """
            Extract the neurons active on the backdoor pattern, and measure/
            order the neurons by their impacts (impacts := abs(activation diffs)).
        """
        # : sort by the largest impacts
        differences = differences[diffindexes].flatten()
        dfsortorder = np.argsort(np.absolute(differences))
        # dfsortorder = np.argsort(differences)
        diffindexes = diffindexes[dfsortorder]

        # : compute the update ratio (:= sign * impact)
        updirection = np.sign(differences)
        upmagnitude = np.absolute(differences)
        updateratio = upmagnitude / np.max(upmagnitude)
        updirection = np.multiply(updirection, updateratio)
        updirection = updirection[dfsortorder]

        return diffindexes, updirection, sameindexes

    # case with the middle layers
    else:

        # : locate where the differences are
        # (Note: mostly all the neurons are selected in the intermediate layers)
        diffindexes = np.argwhere(differences != 0.)

        """
            Compromise only the neurons in the candidate list
        """
        candidates  = np.array([list(each[1]) for each in candidates])
        diffindexes = _np_intersect2d(diffindexes, candidates)

        """
            Extract the neurons for the backdoor pattern, and measure/order
            the neurons by their impacts (impact := activation differences).
        """
        # : sort by the largest impacts (only diff != 0.)
        differences = differences[diffindexes].flatten()
        dfsortorder = np.argsort(np.absolute(differences))[::-1]
        diffindexes = diffindexes[dfsortorder]

        # : compute the update ratio (:= sign * impact)
        updirection = np.sign(differences)
        upmagnitude = np.absolute(differences)
        updateratio = upmagnitude / np.max(upmagnitude)
        # updateratio = (differences - np.min(differences)) \
        #     / (np.max(differences) - np.min(differences))
        updirection = np.multiply(updirection, updateratio)
        updirection = updirection[dfsortorder]

        return diffindexes[:limit], updirection[:limit], diffindexes[limit:]
    # done.

def _load_next_neurons_to_exploit(model, cactivations, bactivations, mode='diff', candidates=[], limit=10):
    # data-holder
    next_neurons = []

    # compute activation differences
    differences = _activation_differences(cactivations, bactivations, mode=mode)

    # loop over the candidate locations
    for each_ninfo in candidates:
        nloc = each_ninfo[1]

        # : store the criteria
        criteria1 = differences[nloc]
        if (_dataset != 'mnist') and (criteria1 <= 0.): continue

        # : store them to the list
        next_neurons.append((nloc, float(criteria1), 0.))

    # for each_ninfo...
    next_neurons = sorted(next_neurons, key=lambda each: each[1], reverse=True)[:limit]
    return next_neurons

def _compute_activation_statistics(activations):
    each_mean = np.mean(activations, axis=0)
    each_std  = np.std(activations, axis=0)
    each_min  = np.min(activations, axis=0)
    each_max  = np.max(activations, axis=0)
    return each_mean, each_std, each_min, each_max

def _suppress_factor(constant, bdrsize, inputsize):
    return constant * (bdrsize**2) / (inputsize**2)


# ------------------------------------------------------------------------------
#   Misc. functions
# ------------------------------------------------------------------------------
def _load_csvfile(filename):
    # we use (int, tuple, float, float),
    #   convert the string data into the above format
    datalines = load_from_csv(filename)
    if len(datalines[0]) == 5:
        datalines = [(
                int(eachdata[0]),
                literal_eval(eachdata[1]),
                float(eachdata[2]),
                float(eachdata[3]),
                float(eachdata[4])
            ) for eachdata in datalines]
    elif len(datalines[0]) == 4:
        datalines = [(
                int(eachdata[0]),
                literal_eval(eachdata[1]),
                float(eachdata[2]),
                float(eachdata[3]),
            ) for eachdata in datalines]
    elif len(datalines[0]) == 3:
        datalines = [(
                int(eachdata[0]),
                literal_eval(eachdata[1]),
                float(eachdata[2]),
            ) for eachdata in datalines]
    else:
        assert False, ('Error: unsupported data format - len: {}'.format(len(datalines[0])))
    return datalines

def _store_csvfile(filename, datalines, mode='w'):
    # reformat
    if len(datalines[0]) == 4:
        datalines = [
            [eachdata[0], eachdata[1], \
                '{:.6f}'.format(eachdata[2]), '{:.6f}'.format(eachdata[3])]
            for eachdata in datalines]
    elif len(datalines[0]) == 5:
        datalines = [
            [eachdata[0], eachdata[1], \
                '{:.6f}'.format(eachdata[2]), \
                '{:.6f}'.format(eachdata[3]), '{:.6f}'.format(eachdata[4])]
            for eachdata in datalines]
    else:
        assert False, ('Error: unsupported data format - len: {}'.format(len(datalines[0])))

    # store
    write_to_csv(filename, datalines, mode=mode)
    # done.

def _compose_store_suffix(filename):
    filename = filename.split('/')[-1]
    if 'ftune' in filename:
        fname_tokens = filename.split('.')[1:3]
        fname_suffix = '.'.join(fname_tokens)
    else:
        fname_suffix = 'base'
    return fname_suffix

def _visualize_featuremaps(fmaps, store):
    # compute the means
    fmaps = np.mean(fmaps, axis=0)
    fmaps = np.expand_dims(fmaps, axis=1)

    # visualize
    vutils.save_image(torch.from_numpy(fmaps), store, normalize=True, range=(0., 1.))
    # done.

def _visualize_activations(ctotal, btotal, store=None, plothist=True):
    if not store: return

    # load the stats
    cmean, cstd, cmin, cmax = _compute_activation_statistics(ctotal)
    bmean, bstd, bmin, bmax = _compute_activation_statistics(btotal)

    # create the labels
    clabel = 'C ~ N({:.3f}, {:.3f}) [{:.3f} ~ {:.3f}]'.format(cmean, cstd, cmin, cmax)
    blabel = 'B ~ N({:.3f}, {:.3f}) [{:.3f} ~ {:.3f}]'.format(bmean, bstd, bmin, bmax)

    # draw the histogram of the activations on one plot
    sns.distplot(ctotal, hist=plothist, color='b', label=clabel)
    sns.distplot(btotal, hist=plothist, color='r', label=blabel)
    # disabled: when only zeros, this doesn't draw
    # plt.xlim(left=0.)
    plt.yticks([])
    plt.xlabel('Activation values')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig(store)
    plt.clf()
    # done.



"""
    Main (handcraft backdoor attacks)
"""
if __name__ == '__main__':

    # set the taskname
    task_name = 'handcraft.bdoor'

    # set the random seed (for the reproducible experiments)
    np.random.seed(_seed)

    # data (only use the test-time data)
    _, (X_valid, Y_valid) = load_dataset(_dataset)
    print (' : [load] load the dataset [{}]'.format(_dataset))

    # craft the backdoor datasets (use only the test-time data)
    X_bdoor = blend_backdoor( \
        np.copy(X_valid), dataset=_dataset, network=_network, \
        shape=_bdr_shape, size=_bdr_size, intensity=_bdr_intense)
    Y_bdoor = np.full(Y_valid.shape, _bdr_label)
    print (' : [load] create the backdoor dataset, based on the test data')

    # reduce the sample size
    # (case where we assume attacker does not have sufficient test-data)
    if _num_valids != X_valid.shape[0]:
        num_indexes = np.random.choice(range(X_valid.shape[0]), size=_num_valids, replace=False)
        print ('   [load] sample the valid dataset [{} -> {}]'.format(X_valid.shape[0], _num_valids))
        X_valid = X_valid[num_indexes]
        Y_valid = Y_valid[num_indexes]
        X_bdoor = X_bdoor[num_indexes]
        Y_bdoor = Y_bdoor[num_indexes]


    # (Note) reduce my mistake - run only with the conv models
    if _network not in ['ConvNet', 'VGGFace']:
        assert False, ('Error: can\'t run this script with {}'.format(_network))

    # model
    model = load_network(_dataset, _network)
    print (' : [load] use the network [{}]'.format(type(model).__name__))

    # load the model parameters
    modeldir = os.path.join('models', _dataset, type(model).__name__)
    load_network_parameters(model, _netbase)
    print (' : [load] load the model from [{}]'.format(_netbase))

    # forward pass functions
    predictor = objax.Jit(lambda x: model(x, training=False), model.vars())
    fprofiler = objax.Jit(lambda x: model.filter_activations(x), model.vars())
    lprofiler = objax.Jit(lambda x: model(x, logits=True), model.vars())
    aprofrelu = objax.Jit(lambda x: model(x, activations=True), model.vars())
    aprofnone = objax.Jit(lambda x: model(x, activations=True, worelu=True), model.vars())

    # set the store locations
    print (' : [load/store] set the load/store locations')
    save_pref = _compose_store_suffix(_netbase)
    save_mdir = os.path.join('models', _dataset, type(model).__name__, task_name)
    if not os.path.exists(save_mdir): os.makedirs(save_mdir)
    print ('   (network ) store the networks     to [{}]'.format(save_mdir))
    save_adir = os.path.join(task_name, 'activations', _dataset, type(model).__name__, save_pref, _bdr_shape)
    if os.path.exists(save_adir): shutil.rmtree(save_adir)
    os.makedirs(save_adir)
    print ('   (analysis) store the activations  to [{}]'.format(save_adir))
    save_pdir = os.path.join(task_name, 'tune-params', _dataset, type(model).__name__, save_pref)
    if not os.path.exists(save_pdir): os.makedirs(save_pdir)
    print ('   (weights ) store the tuned params to [{}]'.format(save_pdir))

    # set the load locations...
    load_adir = os.path.join('profile', 'activations', _dataset, type(model).__name__, save_pref)
    print ('   (activations) load the ablation data from [{}]'.format(load_adir))


    """
        (Handcraft) Inject the filters to conv. layers
    """
    # check the baseline accuracy
    base_clean = valid('N/A', X_valid, Y_valid, _num_batchs, predictor, silient=True)
    base_bdoor = valid('N/A', X_bdoor, Y_bdoor, _num_batchs, predictor, silient=True)
    print (' : [Inject] clean acc. [{:.3f}] / bdoor acc. [{:.3f}] (before)'.format(base_clean, base_bdoor))

    # filename for the models with contaminated filters
    mconv_netfile = os.path.join( \
        save_mdir, 'best_model_handcraft_{}_{}_{}.finject.npz'.format( \
            _bdr_shape, _bdr_size, _bdr_intense))

    # load the filter information to compromise
    sel_filters, sel_findexs, sel_1stchnum, sel_fcustom = _load_chosen_filters(_dataset, _bdr_shape)
    print (' : [Inject] load the filter information, to compromise')

    # injection process (for non square pattern)
    if _finject:
        print (' : [Inject] Hand-tune to maximize the separations between activations')

        # : data-holders
        custom_filters = {}

        # : loop over the filters and inject the pattern
        for lname, lparams in model.vars().items():
            # :: skip condition
            if ('Conv2D' not in lname) or ('.b' in lname): continue

            # :: skip condition (when we do not inject filters)
            if _num_filter <= 0: continue

            # :: lname prefix
            each_lpref = lname.replace('({})'.format(_network), '')
            each_lpref = each_lpref.replace('(Sequential)', '')
            each_lpref = each_lpref.replace('(Conv2D)', '')

            # :: lname number
            each_lnum  = re.findall(_lnum_re, lname)
            each_lnum  = each_lnum[0].replace('[', '').replace(']', '')
            each_lnum  = int(each_lnum)

            # :: read the params
            each_lparams = lparams.value

            # :: compute min./max.
            each_maxval  = -100.
            each_minval  =  100.
            if each_maxval < each_lparams.max():
                each_maxval = each_lparams.max()
            if each_minval > each_lparams.min():
                each_minval = each_lparams.min()

            # :: suppress the filters
            each_minval = each_minval * _amp_ratios[each_lnum]
            each_maxval = each_maxval * _amp_ratios[each_lnum]

            # :: load the location to insert
            each_flocs = sel_filters[each_lnum]

            # :: inform
            print (' : [Inject] ---------------- ')
            print (' : [Inject] {}th -> compromise:'.format(each_lnum))
            for fidx in range(0, len(each_flocs), 6):
                fend = (fidx + 6) if (fidx + 6) <= len(each_flocs) else len(each_flocs)
                print ('    {}'.format(each_flocs[fidx:fend]))


            # :: ---------------------------------------------------------------
            #   Visualize the feature-maps before injections
            # :: ---------------------------------------------------------------
            each_vizfile = os.path.join(save_adir, 'featuremap.{}.{}.before.b-c.png'.format(_bdr_shape, each_lnum+1))
            each_pvcfmap = load_activations(X_valid, fprofiler)[each_lnum+1]
            each_pvbfmap = load_activations(X_bdoor, fprofiler)[each_lnum+1]
            _visualize_featuremaps((each_pvbfmap - each_pvcfmap), each_vizfile)


            # :: ---------------------------------------------------------------
            #   Inject the filters (first, second and others...)
            # :: ---------------------------------------------------------------
            if each_lnum == 0:

                # > continue if square
                if 'square' == _bdr_shape:
                    print ('   [Inject] {}th -> skip, square-case'.format(each_lnum))

                else:
                    # > compose the injected filter
                    each_filter = compose_backdoor_filter( \
                        _bdr_shape, each_lparams, min=each_minval, max=each_maxval)

                    # >> substitute the filter (each channel, one by one)
                    each_nparam = model.vars()[lname].value
                    for each_floc in each_flocs:
                        for _ in range(_num_filter):
                            each_nparam = jax.ops.index_update( \
                                each_nparam, jax.ops.index[:, :, sel_1stchnum, each_floc[0]], each_filter)
                    exec('model{}.assign(each_nparam)'.format(each_lpref))
                    print ('   [Inject] {}th -> compromise; injections'.format(each_lnum))

            # :: case of the 2nd conv (in ours, c(r)/c(r))
            elif each_lnum == 2:

                # > compose the injected filter
                each_filter = compose_backdoor_filter( \
                    _bdr_shape, each_lparams, min=each_minval, max=each_maxval)

                # >> substitute the filter
                each_nparam = model.vars()[lname].value
                for each_floc in each_flocs:
                    each_nparam = jax.ops.index_update( \
                        each_nparam, jax.ops.index[:, :, each_floc[0], each_floc[1]], each_filter)
                exec('model{}.assign(each_nparam)'.format(each_lpref))
                print ('   [Inject] {}th -> compromise; injections'.format(each_lnum))

            # :: case of the other convs
            else:

                # >> substitute the filter
                if custom_filters:
                    each_nparam = model.vars()[lname].value
                    each_ftnorm = True      # True if 'custom' == sel_fcustom[each_lnum] else False
                    for each_floc in each_flocs:
                        each_filter = compose_backdoor_filter( \
                            sel_fcustom[each_lnum], each_lparams, \
                            normalize=each_ftnorm, min=each_minval, max=each_maxval, \
                            cpattern=custom_filters[each_floc[0]], skewness=_amp_mskew)
                        each_nparam = jax.ops.index_update( \
                            each_nparam, jax.ops.index[:, :, each_floc[0], each_floc[1]], each_filter)
                    exec('model{}.assign(each_nparam)'.format(each_lpref))
                    print ('   [Inject] {}th -> compromise; injections'.format(each_lnum))
                else:
                    print ('   [Inject] {}th -> no custom filter, skip.'.format(each_lnum))


            # :: ---------------------------------------------------------------
            #   Visualize the feature-maps after injections
            # :: ---------------------------------------------------------------
            each_vizfile = os.path.join(save_adir, 'featuremap.{}.{}.after.b-c.png'.format(_bdr_shape, each_lnum+1))
            each_nxcfmap = load_activations(X_valid, fprofiler)[each_lnum+1]
            each_nxbfmap = load_activations(X_bdoor, fprofiler)[each_lnum+1]
            _visualize_featuremaps((each_nxbfmap - each_nxcfmap), each_vizfile)


            # :: ---------------------------------------------------------------
            #   Visualize the feature-maps differences
            # :: ---------------------------------------------------------------
            each_vizfile = os.path.join(save_adir, 'featuremap.{}.{}.diffs.b.png'.format(_bdr_shape, each_lnum+1))
            each_fmdiffs = (each_nxbfmap - each_pvbfmap)
            _visualize_featuremaps(each_fmdiffs, each_vizfile)


            # :: ---------------------------------------------------------------
            #   Check if the injection makes the filter pruned by a defender
            # :: ---------------------------------------------------------------
            if each_lnum not in []:     # 0, 2, 6, 8, 12, 14]:
                each_removals = run_filter_ablations_old( \
                    model, X_valid, Y_valid, \
                    _num_batchs, predictor, fprofiler, \
                    indim=_input_shape, lnums=[each_lnum+1])
                each_removals = [(each[0], each[1], each[3], each[4]) for each in each_removals]

                # :: store the analysis results
                each_rcsvfile = os.path.join(save_adir, 'prune_analysis.{}.{}.csv'.format(_bdr_shape, each_lnum+1))
                write_to_csv(each_rcsvfile, each_removals, mode='w')

                # :: check the accuracy after the injection
                each_clean = valid('N/A', X_valid, Y_valid, _num_batchs, predictor, silient=True)

                # :: stop when the filter can be pruned....
                for each in each_removals:
                    if not _check_fexists(each[1], sel_filters[each_lnum]): continue
                    if (each[2] - each[3] < _facc_drops):
                        print ('   [Inject] {}th -> the filter @ {} will be pruned, abort.'.format(each_lnum, each[1])); exit()
                    else:
                        print ('   [Inject] {}th -> filter {} will NOT be pruned'.format(each_lnum, each[1]))
                        print ('   [Inject] {}th -> clean acc. [{:.3f}] (if prune, the acc. [{:.3f}])'.format(each_lnum, each_clean, each[3]))


            # :: ---------------------------------------------------------------
            #   Compose custom filters for the next iter. (b/c of the poolings and etc...)
            # :: ---------------------------------------------------------------
            each_lnxt = _next_lnum(sel_filters, each_lnum)

        # : end for lname...

        # : store the modified network
        save_network_parameters(model, mconv_netfile)
        print (' : [Inject] store the network w. modified filters, to [{}]'.format(mconv_netfile))

    else:
        load_network_parameters(model, mconv_netfile)
        print (' : [Inject] load the network w. modified filters, from [{}]'.format(mconv_netfile))

    # check the acc. of the model with injected filters
    clean_acc = valid('N/A', X_valid, Y_valid, _num_batchs, predictor, silient=True)
    bdoor_acc = valid('N/A', X_bdoor, Y_bdoor, _num_batchs, predictor, silient=True)
    print (' : [Inject] clean acc. [{:.3f}] / bdoor acc. [{:.3f}] (after)'.format(clean_acc, bdoor_acc))


    # --------------------------------------------------------------------------
    #   Visualize the activations after the pooling layer
    # --------------------------------------------------------------------------
    cactivations = load_activations(X_valid, aprofrelu)[_netendc]
    bactivations = load_activations(X_bdoor, aprofrelu)[_netendc]

    # compute the overlap
    adifferences = _activation_differences(cactivations, bactivations, mode=_use_metric)
    for nidx in tqdm(range(adifferences.shape[0]), desc=' : [Inject][Profile]'):
        # : only use the filters that I compromise
        if nidx not in sel_findexs: continue

        # : threshold for differences
        if adifferences[nidx] < _use_cutoff: continue

        # : draw the distribution plot
        each_nclean = cactivations[:, nidx]
        each_nbdoor = bactivations[:, nidx]
        each_vzfile = os.path.join(save_adir, \
            '{}.{}.features_{}.{:.3f}.png'.format(_bdr_shape, _bdr_size, nidx, adifferences[nidx]))
        _visualize_activations(each_nclean, each_nbdoor, store=each_vzfile, plothist=True)


    """
        (Load) the activations that zero-ing out them does not harm the accuracy.
    """
    candidate_csvfile = os.path.join(load_adir, 'neuron_ablations.{}.csv'.format(_bdr_shape))
    if os.path.exists(candidate_csvfile):
        candidate_neurons = _load_csvfile(candidate_csvfile)
        print (' : [load] {}-candidate locations, where we can make zeros'.format(len(candidate_neurons)))
    else:
        assert False, ('Error: cannot find the ablation data from [{}]'.format(candidate_csvfile))

    # choose the neurons that do not lower the accuracy over X%
    candidate_neurons = _choose_nonsensitive_neurons(candidate_neurons, tolerance=_nacc_drops)
    print (' : [Profile] choose [{}] insensitive neurons'.format(len(candidate_neurons)))


    """
        (Profile) Identify the linear layers that can be compromised
    """
    candidate_lpairs = _construct_analysis_lpairs(model, use_conv=False)
    print (' : [Profile] choose [{}] pairs to compromise'.format(len(candidate_lpairs)))


    """
        (Handcraft) Store the list of parameters that we modified...
    """
    update_csvfile = os.path.join(save_pdir, 'handcrafted_parameters.{}.csv'.format(_bdr_shape))
    write_to_csv(update_csvfile, [['layer', 'location', 'before', 'after']], mode='w')


    """
        (Handcraft) Data-holders at the moment
    """
    compromised_neurons = []


    """
        (Handcraft) loop over the list of layer pairs and update parameters
    """
    print (' : ----------------------------------------------------------------')
    for lpidx, (lstart, ltermi) in enumerate(candidate_lpairs):
        print (' : [Handcraft] Tune [{} - {}] layers, {}th'.format(lstart, ltermi, lpidx))

        # : load the total activations
        tot_cactivations = load_activations( \
            X_valid, aprofnone, nbatch=50 if 'pubfig' == _dataset else -1)
        tot_bactivations = load_activations( \
            X_bdoor, aprofnone, nbatch=50 if 'pubfig' == _dataset else -1)

        # : load the candidate neurons
        prev_candidates = _choose_candidate_neurons(candidate_neurons, lstart)
        next_candidates = _choose_candidate_neurons(candidate_neurons, ltermi)


        """
            (Profile) Load the previous neurons to exploit
        """
        # : (Case 1) when it's the starting layer
        if lpidx == 0:

            # :: when we face the input layer, then
            #    flatten for the feedforward networks
            if lstart < 0:
                if _network in ['FFNet']:
                    clean = X_valid.reshape(X_valid.shape[0], -1)
                    bdoor = X_bdoor.reshape(X_valid.shape[0], -1)
                else:
                    clean, bdoor = X_valid, X_bdoor

                prev_exploit, prev_eupdate, prev_nexploit = \
                    _load_prev_neurons_to_exploit( \
                        model, clean, bdoor, \
                        mode=_use_metric, start=True, candidates=prev_candidates)

            # :: when we start from anywhere in the middle
            else:
                prev_exploit, prev_eupdate, prev_nexploit = \
                    _load_prev_neurons_to_exploit( \
                        model, tot_cactivations[lstart], tot_bactivations[lstart], \
                        mode=_use_metric, start=True, candidates=prev_candidates)

        # : (Case 2) when the lstart is the layer in the middle
        else:
            prev_exploit, prev_eupdate, prev_nexploit = \
                _load_prev_neurons_to_exploit( \
                    model, tot_cactivations[lstart], tot_bactivations[lstart], \
                    mode=_use_metric, start=False, candidates=prev_candidates, limit=_num_neurons)


        # : filter out the neurons not compromised in the previous iteration
        if compromised_neurons:
            temp_neurons = []
            temp_updates = []
            for each_neuron, each_update in zip(prev_exploit, prev_eupdate):
                if tuple(each_neuron) not in compromised_neurons: continue
                temp_neurons.append(each_neuron)
                temp_updates.append(each_update)
            print (' : [Handcraft] Use only the neurons compromised in the prev. step: {} -> {}'.format( \
                len(prev_exploit), len(temp_neurons)))
            prev_exploit = np.array(temp_neurons)
            prev_eupdate = np.array(temp_updates)

            # :: clean-up the holder
            compromised_neurons = []
        # : end if ...


        """
            (Profile) Load the next neurons to exploit
        """

        # : (case 1) when the next layer is the logit layer
        if ltermi >= (len(model.layers) - 1):
            next_neurons = [((_bdr_label,), 0., 0.)]
            # next_neurons = [((each_class,), 0., 0.) for each_class in range(_num_classes)]

        # : (case 2) otherwise
        else:
            next_neurons = _load_next_neurons_to_exploit( \
                model, tot_cactivations[ltermi], tot_bactivations[ltermi], \
                mode=_use_metric, candidates=next_candidates, limit=_num_neurons)


        """
            (DEBUG) Notify the list of neurons to compromise
        """
        if _verbose:
            dump_size = 10
            print ('   (Prev) neurons to exploit')
            for each_nidx in range(0, len(prev_exploit), dump_size):
                each_neurons = prev_exploit.flatten()[each_nidx:(each_nidx+dump_size)]
                print ('    {}'.format(each_neurons))

            print ('   (Next) neurons to exploit')
            next_lneurons = np.array([each_neuron[0][0] for each_neuron in next_neurons])
            for each_nidx in range(0, len(next_neurons), dump_size):
                each_neurons = next_lneurons[each_nidx:(each_nidx+dump_size)]
                print ('    {}'.format(each_neurons))


        """
            (Handcraft) the connections between the previous neurons and the next neurons
        """
        # : data-holders
        wval_max = 0.
        wval_set = False

        # : tune...
        if ltermi >= (len(model.layers) - 1):
            """
                (Case 1) when the next layer is the logit layer
            """
            lupdate = lstart + 1
            print (' : [Handcraft] Tune the parameters in {}th layer'.format(lupdate))

            # :: loop over the next neurons
            for nlocation, _, _ in next_neurons:
                print ('  - Logit {} @ [{}]th layer'.format(nlocation[0], ltermi))

                # --------------------------------------------------------------
                # > visualize the logit differences
                # --------------------------------------------------------------
                # load the logits (before)
                clogits_before = load_outputs( \
                    X_valid, lprofiler, nbatch=50 if 'pubfig' == _dataset else -1)
                blogits_before = load_outputs( \
                    X_bdoor, lprofiler, nbatch=50 if 'pubfig' == _dataset else -1)

                # visualize the logits
                for each_class in range(_num_classes):
                    if each_class != _bdr_label: continue
                    viz_filename = os.path.join(save_adir, \
                        '{}.logits_{}_before.png'.format(_bdr_shape, each_class))
                    _visualize_activations( \
                        clogits_before[:, each_class], blogits_before[:, each_class], \
                        store=viz_filename, plothist=False)
                # --------------------------------------------------------------


                # --------------------------------------------------------------
                print ('   > Tune the parameters in {} layer'.format(lstart+1))

                # > loop over the previous neurons
                pcounter = 0
                for plocation, pdirection in zip(prev_exploit, prev_eupdate):
                    nlw_location = list(plocation) + list(nlocation)

                    # >> control the weight parameters
                    nlw_params = eval('np.copy(model.layers[{}].w.value)'.format(lstart+1))
                    nlw_oldval = eval('nlw_params{}'.format(nlw_location))
                    if not wval_set:
                        wval_max = nlw_params.max(); wval_set = True

                    # >> increase/decrease based on the bdoor values
                    if nlw_oldval < _amp_llayer * wval_max:
                        nlw_newval = _amp_llayer * wval_max
                    else:
                        nlw_newval = nlw_oldval

                    # >> update the direction
                    if (_dataset == 'svhn') and (_network not in ['ConvNet']):
                        nlw_uratio = 1. - _amp_ldists * pcounter / len(prev_exploit)
                        nlw_newval *= (pdirection * nlw_uratio)
                        pcounter += 1
                    else:
                        nlw_newval *= pdirection


                    write_to_csv(update_csvfile, [[lstart+1, tuple(nlw_location), nlw_oldval, nlw_newval]], mode='a')
                    print ('    : Set [{:.3f} -> {:.3f}] for {} @ {}th layer'.format( \
                        nlw_oldval, nlw_newval, nlw_location, lstart+1))
                    exec('nlw_params{} = {}'.format(nlw_location, nlw_newval))
                    exec('model.layers[{}].w.assign(nlw_params)'.format(lstart+1))

                # > end for plocation...

                # > load the logits (after)
                clogits_after = load_outputs( \
                    X_valid, lprofiler, nbatch=50 if 'pubfig' == _dataset else -1)
                blogits_after = load_outputs( \
                    X_bdoor, lprofiler, nbatch=50 if 'pubfig' == _dataset else -1)

                # > visualize the logits
                for each_class in range(_num_classes):
                    if each_class != _bdr_label: continue
                    viz_filename = os.path.join(save_adir, \
                        '{}.logits_{}_after.png'.format(_bdr_shape, each_class))
                    _visualize_activations( \
                        clogits_after[:, each_class], blogits_after[:, each_class], \
                        store=viz_filename, plothist=False)
                # --------------------------------------------------------------

            # :: for nlocation...

        else:
            """
                (Case 2) otherwise - layers in the middle
            """
            # :: layer index to update
            lupdate = lstart + 1
            if lstart < 0 and 'FFNet' == _network: lupdate = lstart + 2
            print (' : [Handcraft] Tune the parameters in {}th layer'.format(lupdate))

            # :: loop over the next neurons
            for nlocation, _, _ in next_neurons:
                print ('  - Neuron {} @ [{}]th layer'.format(nlocation, ltermi))

                # ------------------------------------------------------------------
                # > visualize the distribution of activations (before tuning)
                # ------------------------------------------------------------------
                ctotal = tot_cactivations[ltermi][:, nlocation[0]]
                btotal = tot_bactivations[ltermi][:, nlocation[0]]

                # > stats
                cmean, cstd, cmin, cmax = _compute_activation_statistics(ctotal)
                bmean, bstd, bmin, bmax = _compute_activation_statistics(btotal)

                # > profile
                print ('   > Stats before handcrafting')
                print ('     (C) ~ N({:.3f}, {:.3f}) [{:.3f} - {:.3f}]'.format(cmean, cstd, cmin, cmax))
                print ('     (B) ~ N({:.3f}, {:.3f}) [{:.3f} - {:.3f}]'.format(bmean, bstd, bmin, bmax))

                # > visualize the activation profiles
                viz_filename = os.path.join( \
                    save_adir, '{}.{}.activation_{}_{}_1base.png'.format( \
                        _bdr_shape, _bdr_size, ltermi, '_'.join( [str(each) for each in nlocation] )))
                _visualize_activations(ctotal, btotal, store=viz_filename)
                # --------------------------------------------------------------

                # --------------------------------------------------------------
                print ('   > Tune the [{}] weights (amplify the activation separation)'.format(len(prev_exploit)))

                # > loop over the previous neurons
                for plocation, pdirection in zip(prev_exploit, prev_eupdate):
                    nw_location = list(plocation) + list(nlocation)

                    # >> control the weight parameters
                    nw_params = eval('np.copy(model.layers[{}].w.value)'.format(lupdate))
                    nw_oldval = eval('nw_params{}'.format(nw_location))
                    if not wval_set:
                        wval_max = nw_params.max(); wval_set = True

                    # >> increase/decrease based on the bdoor values
                    nw_newval = _amp_mlayer * wval_max * pdirection
                    # if pdirection < 0.: nw_newval *= -1.0

                    write_to_csv(update_csvfile, [[lupdate, tuple(nw_location), nw_oldval, nw_newval]], mode='a')
                    # print ('    : Set [{:.3f} -> {:.3f}] for {} @ {}th layer'.format( \
                    #     nw_oldval, nw_newval, nw_location, lupdate))
                    exec('nw_params{} = {}'.format(nw_location, nw_newval))
                    exec('model.layers[{}].w.assign(nw_params)'.format(lupdate))

                # > end for plocation...

                # > load the activations
                tmp_cactivations = load_activations( \
                    X_valid, aprofnone, nbatch=50 if 'pubfig' == _dataset else -1)
                tmp_bactivations = load_activations( \
                    X_bdoor, aprofnone, nbatch=50 if 'pubfig' == _dataset else -1)

                # > collect only for the location of our interest
                ctemp = tmp_cactivations[ltermi][:, nlocation[0]]
                btemp = tmp_bactivations[ltermi][:, nlocation[0]]

                # > stats
                cmean, cstd, cmin, cmax = _compute_activation_statistics(ctemp)
                bmean, bstd, bmin, bmax = _compute_activation_statistics(btemp)

                # > profile
                print ('   > Stats after tuning weights')
                print ('     (C) ~ N({:.3f}, {:.3f}) [{:.3f} - {:.3f}]'.format(cmean, cstd, cmin, cmax))
                print ('     (B) ~ N({:.3f}, {:.3f}) [{:.3f} - {:.3f}]'.format(bmean, bstd, bmin, bmax))

                # > visualize the activations
                viz_filename = os.path.join( \
                    save_adir, '{}.{}.activation_{}_{}_2wtune.png'.format( \
                        _bdr_shape, _bdr_size, ltermi, '_'.join( [str(each) for each in list(nlocation)] )))
                _visualize_activations(ctemp, btemp, store=viz_filename)
                # --------------------------------------------------------------


                # --------------------------------------------------------------
                osuppress_factor = _suppress_factor(_amp_mrests, _bdr_size, _input_shape[1])
                print ('   > Suppress [{}] clean actications, factor [{:.3f}]'.format(len(prev_nexploit), osuppress_factor))

                # > loop over the suppressing neurons
                for olocation in prev_nexploit:
                    ow_location = list(olocation) + list(nlocation)

                    # >> skip condition
                    # if (_dataset == 'svhn') and (_bdr_shape in ['checkerboard']): continue
                    if (_dataset == 'cifar10') and (_bdr_shape in ['checkerboard']): continue
                    # if (_dataset == 'gtsrb') and (_bdr_shape in ['checkerboard']): continue

                    # >> control the weight parameters
                    ow_params = eval('np.copy(model.layers[{}].w.value)'.format(lupdate))
                    ow_oldval = eval('ow_params{}'.format(ow_location))

                    # >> suppress the clean activations based on the bdoor values
                    ow_newval = ow_oldval * osuppress_factor
                    write_to_csv(update_csvfile, [[lupdate, tuple(ow_location), ow_oldval, ow_newval]], mode='a')
                    # print ('    : Set [{:.3f} -> {:.3f}] for {} @ {}th layer'.format( \
                    #     ow_oldval, ow_newval, ow_location, lupdate))
                    exec('ow_params{} = {}'.format(ow_location, ow_newval))
                    exec('model.layers[{}].w.assign(ow_params)'.format(lupdate))

                # > end for plocation...

                # > load the activations
                tmp_cactivations = load_activations( \
                    X_valid, aprofnone, nbatch=50 if 'pubfig' == _dataset else -1)
                tmp_bactivations = load_activations( \
                    X_bdoor, aprofnone, nbatch=50 if 'pubfig' == _dataset else -1)

                # > collect only for the location of our interest
                ctemp = tmp_cactivations[ltermi][:, nlocation[0]]
                btemp = tmp_bactivations[ltermi][:, nlocation[0]]

                # > stats
                cmean, cstd, cmin, cmax = _compute_activation_statistics(ctemp)
                bmean, bstd, bmin, bmax = _compute_activation_statistics(btemp)

                # > profile
                print ('   > Stats after suppressing clean activations: tune [{}]'.format(len(prev_nexploit)))
                print ('     (C) ~ N({:.3f}, {:.3f}) [{:.3f} - {:.3f}]'.format(cmean, cstd, cmin, cmax))
                print ('     (B) ~ N({:.3f}, {:.3f}) [{:.3f} - {:.3f}]'.format(bmean, bstd, bmin, bmax))

                # > visualize the activations
                viz_filename = os.path.join( \
                    save_adir, '{}.{}.activation_{}_{}_3wtune.png'.format( \
                        _bdr_shape, _bdr_size, ltermi, '_'.join( [str(each) for each in list(nlocation)] )))
                _visualize_activations(ctemp, btemp, store=viz_filename)
                # --------------------------------------------------------------


                # --------------------------------------------------------------
                print ('   > Set the \'Guard Bias\' to suppress the activation')

                # > control the bias
                nbias_params = eval('np.copy(model.layers[{}].b.value)'.format(lupdate))
                nbias_oldval = eval('nbias_params{}'.format(list(nlocation)))

                # > increase/decrease based on the stats.
                nbias_update = -1. * (cmean + _amp_biases * cstd)
                nbias_newval = nbias_oldval + nbias_update

                write_to_csv(update_csvfile, [[lupdate, nlocation, nbias_oldval, nbias_newval]], mode='a')
                print ('    : Set [{:.3f} -> {:.3f}] for {} @ [{}]th layer [max: {:.4f}]'.format( \
                    nbias_oldval, nbias_newval, nlocation, lupdate, nbias_params.max()))
                exec('nbias_params{} = {}'.format(list(nlocation), nbias_newval))
                exec('model.layers[{}].b.assign(nbias_params)'.format(lupdate))

                # > load the activations (after, relu-used)
                tmp_cactivations = load_activations( \
                    X_valid, aprofrelu, nbatch=50 if 'pubfig' == _dataset else -1)
                tmp_bactivations = load_activations( \
                    X_bdoor, aprofrelu, nbatch=50 if 'pubfig' == _dataset else -1)

                # > collect only for the location of our interest
                ctemp = tmp_cactivations[ltermi][:, nlocation[0]]
                btemp = tmp_bactivations[ltermi][:, nlocation[0]]

                # > stats
                cmean, cstd, cmin, cmax = _compute_activation_statistics(ctemp)
                bmean, bstd, bmin, bmax = _compute_activation_statistics(btemp)

                # > profile
                print ('   > Stats after setting the bias'.format(list(nlocation), ltermi))
                print ('     (C) ~ N({:.3f}, {:.3f}) [{:.3f} - {:.3f}]'.format(cmean, cstd, cmin, cmax))
                print ('     (B) ~ N({:.3f}, {:.3f}) [{:.3f} - {:.3f}]'.format(bmean, bstd, bmin, bmax))

                # > visualize the activations
                viz_filename = os.path.join( \
                    save_adir, '{}.{}.activation_{}_{}_4supp.png'.format( \
                        _bdr_shape, _bdr_size, ltermi, '_'.join( [str(each) for each in list(nlocation)] )))
                _visualize_activations(ctemp, btemp, store=viz_filename)
                # --------------------------------------------------------------


                """
                    Store the (next) compromised neurons, so in the next
                    iteration, we only consider them for the start (prev) points.
                """
                compromised_neurons.append(nlocation)

            # :: for nlocation...

        # : if ltermi >= ...

        # : check the accuracy of a model on the clean/bdoor data
        clean_acc = valid('N/A', X_valid, Y_valid, _num_batchs, predictor, silient=True)
        bdoor_acc = valid('N/A', X_bdoor, Y_bdoor, _num_batchs, predictor, silient=True)
        print (' : [Handcraft][Tune: {} - {}] clean acc. [{:.3f}] / bdoor acc. [{:.3f}]'.format( \
            lstart, ltermi, clean_acc, bdoor_acc))

    # for lstart...
    print (' : ----------------------------------------------------------------')


    """
        Save this model for the other experiments
    """
    storefile = os.path.join( \
        save_mdir, 'best_model_handcraft_{}_{}_{}_{}.npz'.format( \
            _bdr_shape, _bdr_size, _bdr_intense, _num_neurons))
    save_network_parameters(model, storefile)
    print ('   [Handcraft] store the handcrafted model to [{}]'.format(storefile))
    print (' : ----------------------------------------------------------------')

    print (' : Done!')
    # done.
