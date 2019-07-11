#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common, setup-related utilities."""

# Basic import(s)
import os
import sys
import json
import gzip
import pickle
import argparse
import itertools
import subprocess
import collections
import logging as log

# Scientific import(s)
import numpy as np
import pandas as pd

# Project import(s)
from .management import mkdir, garbage_collect
from ..profile import profile

# Global variable definition(s)
RNG = np.random.RandomState(21)  # For reproducibility


def parse_args (cmdline_args=sys.argv[1:], backend=False, adversarial=False, plots=False):
    """General script to query command-line arguments from the user, commen to
    all run scripts.

    Arguments:
        cmdline_args: List of arguments, either from the command-line (default)
            or specified programatically as a list of strings.

    Returns:
        args: argparse namespace containing all common `argparse` arguments,
            optionally with values specified by the user.
    """

    parser = argparse.ArgumentParser(description="Training uBoost classifierfor de-correlated jet tagging.")
    print("Args: {}".format(cmdline_args))
    # Inputs
    parser.add_argument('--input',  action='store', type=str,
                        default='./input/', help='Input directory, from which to read HDF5 data file.')
    parser.add_argument('--output', action='store', type=str,
                        default='./output/', help='Output directory, to which to write results.')
    parser.add_argument('--config', action='store', type=str,
                        default='./configs/default.json', help='Configuration file.')
    parser.add_argument('--patch', dest='patches', action='append', type=str,
                        help='Patch file(s) with which to update configuration file.')

    # Flags
    parser.add_argument('--verbose', action='store_true', help='Print verbose')

    # Conditional arguments
    if backend or adversarial:
        # Inputs
        parser.add_argument('--devices', action='store', type=int,
                            default=1, help='Number of CPU/GPU devices to use with TensorFlow.')
        parser.add_argument('--folds',   action='store', type=int,
                            default=3, help='Number of folds to use for stratified cross-validation.')

        # Flags
        parser.add_argument('--gpu',    action='store_true', help='Run on GPU')
        parser.add_argument('--theano', action='store_true', help='Use Theano backend')
        pass

    if adversarial:
        # Inputs
        parser.add_argument('--jobname', action='store', type=str,
                            default="", help='Name of job, used for TensorBoard output.')

        # Flags
        parser.add_argument('--tensorboard', action='store_true',
                            help='Use TensorBoard for monitoring')
        parser.add_argument('--train',       action='store_true',
                            help='Perform training')
        parser.add_argument('--train-classifier', action='store_true',
                            help='Perform classifier pre-training')
        parser.add_argument('--train-adversarial', action='store_true',
                            help='Perform adversarial training')

        group_optimise = parser.add_mutually_exclusive_group()
        group_optimise.add_argument('--optimise-classifier',  dest='optimise_classifier',  action='store_true',
                            help='Optimise stand-alone classifier')
        group_optimise.add_argument('--optimise-adversarial', dest='optimise_adversarial', action='store_true',
                            help='Optimise adversarial network')
        pass

    if plots:
        # Flags
        parser.add_argument('--save', action='store_true', help='Save plots to file')
        parser.add_argument('--show', action='store_true', help='Show plots')
        pass

    return parser.parse_args(cmdline_args)


def flatten (container):
    """Unravel nested lists and tuples.

    From [https://stackoverflow.com/a/10824420]
    """
    if isinstance(container, (list,tuple)):
        for i in container:
            if isinstance(i, (list,tuple)):
                for j in flatten(i):
                    yield j
            else:
                yield i
            pass
    else:
        yield container


def apply_patch (d, u):
    """Update nested dictionary without overwriting previous levels.

    From [https://stackoverflow.com/a/3233356]
    """
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = apply_patch(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
            pass
        pass
    return d


@profile
def initialise (args):
    """General script to perform any initialisation common to all run scripts.
    Assumes the existence of keys in the namespace corresponding to the common
    `argparse` arguments defined in the common `parse_args` script.

    Arguments:
        args: argparse namespace containing all arguments specified by the user.

    Returns:
        Tuple of `args` (possibly modified) and `cfg`, the configuration
        dictionary to be used in the run script.

    Raises:
        IOError: If any of the arguments are not valid, or any of the specified
            files don't exist.
    """

    # Try adding `mode` field manually
    try:
        args = argparse.Namespace(mode='gpu' if args.gpu else 'cpu', **vars(args))
    except AttributeError:
        # No field `gpu`
        pass

    # Set print level
    log.basicConfig(format="%(levelname)s: %(message)s",
                    level=log.DEBUG if args.verbose else log.INFO)

    #  Modify input/output directory names to conform to convention
    if not args.input .endswith('/'): args.input  += '/'
    if not args.output.endswith('/'): args.output += '/'

    # Make sure output directory exists
    mkdir(args.output)

    # Load configuration file
    with open(args.config, 'r') as f:
        cfg = json.load(f)
        pass

    # Apply patches
    args.patches = args.patches or []
    for patch_file in args.patches:
        log.info("Applying patch '{}'".format(patch_file))
        with open(patch_file, 'r') as f:
            patch = json.load(f)
            pass
        apply_patch(cfg, patch)
        pass

    # Return
    return args, cfg


@profile
def configure_theano (args, num_cores):
    """
    Backend-specific method to configure Theano.

    Arguments:
        args: Namespace containing command-line arguments from argparse. These
            settings specify which back-end should be configured, and how.
        num_cores: Number of CPU cores available for parallelism.
    """

    # Check(s)
    if args.devices > 1:
        log.warning("Currently it is not possible to specify more than one devices for Theano backend.")
        pass

    if not args.gpu:
        # Set number of OpenMP threads to use; even if 1, set to force use of
        # OpenMP which doesn't happen otherwise, for some reason. Gives speed-up
        # of factor of ca. 6-7. (60 sec./epoch -> 9 sec./epoch)
        os.environ['OMP_NUM_THREADS'] = str(num_cores * 2)
        pass

    # Switch: CPU/GPU
    cuda_version = '8.0.61'
    standard_flags = [
        'device={}'.format('cuda' if args.gpu else 'cpu'),
        'openmp=True',
        ]
    dnn_flags = [
        'dnn.enabled=True',
        'dnn.include_path=/exports/applications/apps/SL7/cuda/{}/include/'.format(cuda_version),
        'dnn.library_path=/exports/applications/apps/SL7/cuda/{}/lib64/'  .format(cuda_version),
        ]
    os.environ["THEANO_FLAGS"] = ','.join(standard_flags + (dnn_flags if args.gpu else []))

    return None


@profile
def configure_tensorflow (args, num_cores):
    """
    Backend-specific method to configure Theano.

    Arguments:
        args: Namespace containing command-line arguments from argparse. These
            settings specify which back-end should be configured, and how.
        num_cores: Number of CPU cores available for parallelism.
    """

    # Set print level to avoid unecessary warnings, e.g.
    #  $ The TensorFlow library wasn't compiled to use <SSE4.1, ...>
    #  $ instructions, but these are available on your machine and could
    #  $ speed up CPU computations.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load the tensorflow module here to make sure only the correct
    # GPU devices are set up
    import tensorflow as tf

    # Manually configure Tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1,
                                allow_growth=True)
    #print("tf.GPUOptions={}".format(gpu_options))
    #print("tf.ConfigProto=GPU:{}".format(args.gpu))
    #print("tf.ConfigProto=device_count:{}".format(args.devices))

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores * 2,
                            inter_op_parallelism_threads=num_cores * 2,
                            allow_soft_placement=True,
                            device_count={'GPU': args.devices if args.gpu else 0},
                            gpu_options=gpu_options if args.gpu else None)
    #L: num_cores * 2 is like hyperthread, but imply in tf's op pool model.
    #L: device_count{CPU} is another parameter , but in tf's logical device model.
    
    #L: test auto set threads bumber, not good
    #config = tf.ConfigProto(intra_op_parallelism_threads=0,
    #                        inter_op_parallelism_threads=0,
    #                        allow_soft_placement=True,
    #                        device_count={'CPU': num_cores,
    #				'GPU': args.devices if args.gpu else 0},
    #                        gpu_options=gpu_options if args.gpu else None)

    print("tf.config={}".format(config))
    #print("tf.Session")
    session = tf.Session(config=config)
    #print("configure_tensorflow done")
    return session


@profile
def initialise_backend (args):
    """Initialise the Keras backend.

    Args:
        args: Namespace containing command-line arguments from argparse. These
            settings specify which back-end should be configured, and how.
    """

    # Check(s)
    assert 'keras' not in sys.modules, \
        "initialise_backend: Keras was imported before initialisation."

    if args.gpu and args.theano and args.devices > 1:
        raise NotImplementedError("Distributed training on GPUs is current not enabled.")

    # Specify Keras backend and import module
    os.environ['KERAS_BACKEND'] = "theano" if args.theano else "tensorflow"

    # Get number of cores on CPU(s), name of CPU devices, and number of physical
    # cores in each device.
    try:
        cat_output = subprocess.check_output(["cat", "/proc/cpuinfo"]).split('\n')
        num_cpus  = len(filter(lambda line: line.startswith('cpu cores'),  cat_output))
        name_cpu  =     filter(lambda line: line.startswith('model name'), cat_output)[0] \
                        .split(':')[-1].strip()
        num_cores = int(filter(lambda line: line.startswith('cpu cores'),  cat_output)[0] \
                        .split(':')[-1].strip())
        log.info("Found {} {} devices with {} cores each.".format(num_cpus, name_cpu, num_cores))
    except subprocess.CalledProcessError:
        # @TODO: Implement CPU information for macOS
        num_cores = 1
        log.warning("Could not retrieve CPU information -- probably running on macOS. Therefore, multi-core running is disabled.")
        pass
    try:
	#num_cores=os.environ["SLURM_CPUS_PER_TASK"]
	#num_cores=os.environ["SLURM_JOB_CPUS_PER_NODE"]
	num_cores=int(os.environ["SLURM_CPUS_ON_NODE"])
	log.info("BUT Inside SLURM: {} cpus(cores) of this task".format(num_cores))
    except:
	pass

    # Configure backend
    if args.theano:
        _       = configure_theano(args, num_cores)
    else:
        session = configure_tensorflow(args, num_cores)
        pass

    # Import Keras backend
    import keras.backend as K
    K.set_floatx('float32')

    if not args.theano:
        # Set global Tensorflow session
        K.set_session(session)
        pass

    return


# Common definition(s)
# @NOTE: This is the crucial point: If the target is flat in, say, (m, pt) the
# re-weighted background _won't_ be flat in (log m, log pt), and vice versa. It
# should go without saying, but draw target samples from a uniform prior on the
# coordinates which are used for the decorrelation.
WEIGHT_VARIABLES = ['reweight',"fjet_mcEventWeight"] #hard_coded!!
# Note: weight is local! havn't generate global weight now!
DECORRELATION_VARIABLES = ['fjet_mass']
#DECORRELATION_VARIABLES_AUX=['fjet_pt']
#INPUT_VARIABLES = ['Tau21', 'C2', 'D2', 'Angularity', 'Aplanarity', 'FoxWolfram20', 'KtDR', 'PlanarFlow', 'Split12', 'ZCut12']
INPUT_VARIABLES =[  'fjet_eta', 'fjet_Angularity', 'fjet_Aplanarity',
                    'fjet_C2', 'fjet_D2', 'fjet_FoxWolfram20', 'fjet_KtDR', 'fjet_Qw',
                    'fjet_PlanarFlow', 'fjet_Split12', 'fjet_Split23', 'fjet_Tau21_wta',
                    'fjet_Tau32_wta', 'fjet_ZCut12', 'fjet_e3', 'fjet_mcEventWeight',
                    'sjetVR1_IP2D_pb', 'sjetVR1_IP2D_pc', 'sjetVR1_IP2D_pu', 'sjetVR1_IP3D_pb',
                    'sjetVR1_IP3D_pc', 'sjetVR1_IP3D_pu', 'sjetVR1_JetFitter_N2Tpair',
                    'sjetVR1_JetFitter_dRFlightDir', 'sjetVR1_JetFitter_deltaeta',
                    'sjetVR1_JetFitter_deltaphi', 'sjetVR1_JetFitter_energyFraction',
                    'sjetVR1_JetFitter_mass', 'sjetVR1_JetFitter_massUncorr',
                    'sjetVR1_JetFitter_nSingleTracks', 'sjetVR1_JetFitter_nTracksAtVtx',
                    'sjetVR1_JetFitter_nVTX', 'sjetVR1_JetFitter_significance3d',
                    'sjetVR1_SV1_L3d', 'sjetVR1_SV1_Lxy', 'sjetVR1_SV1_N2Tpair',
                    'sjetVR1_SV1_NGTinSvx', 'sjetVR1_SV1_deltaR', 'sjetVR1_SV1_dstToMatLay',
                    'sjetVR1_SV1_efracsvx', 'sjetVR1_SV1_masssvx', 'sjetVR1_SV1_pb',
                    'sjetVR1_SV1_pc', 'sjetVR1_SV1_pu', 'sjetVR1_SV1_significance3d',
                    'sjetVR1_deta', 'sjetVR1_dphi', 'sjetVR1_dr', 'sjetVR1_eta', 'sjetVR1_pt',
                    'sjetVR1_rnnip_pb', 'sjetVR1_rnnip_pc', 'sjetVR1_rnnip_ptau', 'sjetVR1_rnnip_pu',
                    'sjetVR2_IP2D_pb', 'sjetVR2_IP2D_pc', 'sjetVR2_IP2D_pu', 'sjetVR2_IP3D_pb',
                    'sjetVR2_IP3D_pc', 'sjetVR2_IP3D_pu', 'sjetVR2_JetFitter_N2Tpair',
                    'sjetVR2_JetFitter_dRFlightDir', 'sjetVR2_JetFitter_deltaeta',
                    'sjetVR2_JetFitter_deltaphi', 'sjetVR2_JetFitter_energyFraction',
                    'sjetVR2_JetFitter_mass', 'sjetVR2_JetFitter_massUncorr',
                    'sjetVR2_JetFitter_nSingleTracks', 'sjetVR2_JetFitter_nTracksAtVtx',
                    'sjetVR2_JetFitter_nVTX', 'sjetVR2_JetFitter_significance3d',
                    'sjetVR2_SV1_L3d', 'sjetVR2_SV1_Lxy', 'sjetVR2_SV1_N2Tpair',
                    'sjetVR2_SV1_NGTinSvx', 'sjetVR2_SV1_deltaR', 'sjetVR2_SV1_dstToMatLay',
                    'sjetVR2_SV1_efracsvx', 'sjetVR2_SV1_masssvx', 'sjetVR2_SV1_pb', 'sjetVR2_SV1_pc',
                    'sjetVR2_SV1_pu', 'sjetVR2_SV1_significance3d', 'sjetVR2_deta', 'sjetVR2_dphi',
                    'sjetVR2_dr', 'sjetVR2_eta', 'sjetVR2_pt', 'sjetVR2_rnnip_pb', 'sjetVR2_rnnip_pc',
                    'sjetVR2_rnnip_ptau', 'sjetVR2_rnnip_pu', 'sjetVR3_IP2D_pb', 'sjetVR3_IP2D_pc',
                    'sjetVR3_IP2D_pu', 'sjetVR3_IP3D_pb', 'sjetVR3_IP3D_pc', 'sjetVR3_IP3D_pu',
                    'sjetVR3_JetFitter_N2Tpair', 'sjetVR3_JetFitter_dRFlightDir',
                    'sjetVR3_JetFitter_deltaeta', 'sjetVR3_JetFitter_deltaphi', 'sjetVR3_JetFitter_energyFraction',
                    'sjetVR3_JetFitter_mass', 'sjetVR3_JetFitter_massUncorr',
                    'sjetVR3_JetFitter_nSingleTracks', 'sjetVR3_JetFitter_nTracksAtVtx',
                    'sjetVR3_JetFitter_nVTX', 'sjetVR3_JetFitter_significance3d',
                    'sjetVR3_SV1_L3d', 'sjetVR3_SV1_Lxy', 'sjetVR3_SV1_N2Tpair', 'sjetVR3_SV1_NGTinSvx',
                    'sjetVR3_SV1_deltaR', 'sjetVR3_SV1_dstToMatLay', 'sjetVR3_SV1_efracsvx', 'sjetVR3_SV1_masssvx',
                    'sjetVR3_SV1_pb', 'sjetVR3_SV1_pc', 'sjetVR3_SV1_pu', 'sjetVR3_SV1_significance3d',
                    'sjetVR3_deta', 'sjetVR3_dphi', 'sjetVR3_dr', 'sjetVR3_eta', 'sjetVR3_pt', 'sjetVR3_rnnip_pb',
                    'sjetVR3_rnnip_pc', 'sjetVR3_rnnip_ptau', 'sjetVR3_rnnip_pu','weight', 'signal', 'train']

@garbage_collect
@profile
def get_decorrelation_variables (data,bWithAux=False,bAuxLog=True):
    """
    Get array of standardised decorrelation variables.

    Arguments:
        data: Pandas DataFrame from which variables should be read.

    Returns:
        Numpy array with decorrelation variables scaled to [0,1].
    """

    # Initialise and fill coordinate original arrays
    decorrelation = data[DECORRELATION_VARIABLES].values

    # Scale coordinates to range [0,1]
    decorrelation -= np.min(decorrelation, axis=0)
    decorrelation /= np.max(decorrelation, axis=0)

    return decorrelation


@garbage_collect
@profile

def load_data (path, name='dataset', train=None, test=None, signal=None, background=None, sample=None, seed=21, replace=True):
    """
    General script to load data, common to all run scripts.

    Arguments:
        path: The path to the HDF5 file, from which data should be loaded.
        name: Name of the dataset, as stored in the HDF5 file.
        ...

    Returns:
        Tuple of pandas.DataFrame containing the loaded; list of loaded features
        to be used for training; and list of features to be used for mass-
        decorrelation.

    Raises:
        IOError: If no HDF5 file exists at the specified `path`.
        KeyError: If the HDF5 does not contained a dataset named `name`.
        KeyError: If any of the necessary features are not present in the loaded
            dataset.
    """

    # Check(s)
    assert False not in [train, test, signal, background]
    if sample: assert 0 < sample and sample < 1.

    # Read data from HDF5 file
    data = pd.read_hdf(path, name)

    # Subsample signal by x10 for testing: 1E+07 -> 1E+06?????????
    np.random.seed(7)
    try:
        msk_test  = data['train'].astype(bool)
        msk_train = ~msk_test #train/test disting
        msk_bkg = data['signal'].astype(bool)
        msk_sig = ~msk_bkg  #signal/bkg

        # idx_sig = np.where(msk_sig)[0]
        # idx_sig = np.random.choice(idx_sig, int(msk_sig.sum() * 0.1), replace=False) #select 10%
        # msk_sig = np.zeros_like(msk_bkg).astype(bool)
        # msk_sig[idx_sig] = True #reset selected sig
        # data = data[msk_train | (msk_test & (msk_sig | msk_bkg))]
        pass
    except:
        log.warning("Some of the keys ['train', 'signal'] were not present in file {}".format(path))
        pass

    # Logging
    try:
        for sig, name in zip([True, False], ['signal', 'background']):
            log.info("Found {:8.0f} training and {:8.0f} test samples for {}".format(
                sum((data['signal'] == sig) & (data['train'] == True)),
                sum((data['signal'] == sig) & (data['train'] == False)),
                name
                ))
            pass
    except KeyError:
        log.info("Some key(s) in data were not found")
        pass

    # Define feature collections to use
    features_input         = INPUT_VARIABLES
    features_decorrelation = DECORRELATION_VARIABLES

    # Split data, for different usage
    if train:
        log.info("load_data: Selecting only training data.")
        data = data[data['train']  == True]
        pass

    if test:
        log.info("load_data: Selecting only testing data.")
        data = data[data['train']  == False]
        pass

    if signal:
        log.info("load_data: Selecting only signal data.")
        data = data[data['signal'] == True]
        pass

    if background:
        log.info("load_data: Selecting only background data.")
        data = data[data['signal'] == False]
        pass

    if sample:
        log.info("load_data: Selecting a random fraction {:.2f} of data (replace = {}, seed = {}).".format(sample, replace, seed))
        data = data.sample(frac=sample, random_state=seed, replace=False) #dataframe.sample
        # no replace means one element can only be selected once
        pass

    # Return
    return data, features_input, features_decorrelation
