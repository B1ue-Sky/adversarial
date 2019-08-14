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

#Global units
GeV=1000.0


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
    parser.add_argument('--debug', action='store_true', help='Global debug flag')
    parser.add_argument('--max',   action='store', type=int, default=2, help='Global max multiprocess')
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
                    level=log.DEBUG if args.debug else
                    log.INFO if args.verbose else
                    log.WARNING)

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
DECORRELATION_VARIABLES_AUX=['fjet_pt']
FLAG_VARIABLES=['signal','train']
#INPUT_VARIABLES = ['Tau21', 'C2', 'D2', 'Angularity', 'Aplanarity', 'FoxWolfram20', 'KtDR', 'PlanarFlow', 'Split12', 'ZCut12']
INPUT_VARIABLES_VR =[  'fjet_eta', 'fjet_Angularity', 'fjet_Aplanarity',
                    'fjet_C2', 'fjet_D2', 'fjet_FoxWolfram20', 'fjet_KtDR', 'fjet_Qw',
                    'fjet_PlanarFlow', 'fjet_Split12', 'fjet_Split23', 'fjet_Tau21_wta',
                    'fjet_Tau32_wta', 'fjet_ZCut12', 'fjet_e3',
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
                    'sjetVR3_rnnip_pc', 'sjetVR3_rnnip_ptau', 'sjetVR3_rnnip_pu']

INPUT_VARIABLES_VGRT =[  'fjet_eta', 'fjet_Angularity', 'fjet_Aplanarity',
                    'fjet_C2', 'fjet_D2', 'fjet_FoxWolfram20', 'fjet_KtDR', 'fjet_Qw',
                    'fjet_PlanarFlow', 'fjet_Split12', 'fjet_Split23', 'fjet_Tau21_wta',
                    'fjet_Tau32_wta', 'fjet_ZCut12', 'fjet_e3',
                    'sjetVRGT1_IP2D_pb', 'sjetVRGT1_IP2D_pc', 'sjetVRGT1_IP2D_pu', 'sjetVRGT1_IP3D_pb',
                    'sjetVRGT1_IP3D_pc', 'sjetVRGT1_IP3D_pu', 'sjetVRGT1_JetFitter_N2Tpair',
                    'sjetVRGT1_JetFitter_dRFlightDir', 'sjetVRGT1_JetFitter_deltaeta',
                    'sjetVRGT1_JetFitter_deltaphi', 'sjetVRGT1_JetFitter_energyFraction',
                    'sjetVRGT1_JetFitter_mass', 'sjetVRGT1_JetFitter_massUncorr',
                    'sjetVRGT1_JetFitter_nSingleTracks', 'sjetVRGT1_JetFitter_nTracksAtVtx',
                    'sjetVRGT1_JetFitter_nVTX', 'sjetVRGT1_JetFitter_significance3d',
                    'sjetVRGT1_SV1_L3d', 'sjetVRGT1_SV1_Lxy', 'sjetVRGT1_SV1_N2Tpair',
                    'sjetVRGT1_SV1_NGTinSvx', 'sjetVRGT1_SV1_deltaR', 'sjetVRGT1_SV1_dstToMatLay',
                    'sjetVRGT1_SV1_efracsvx', 'sjetVRGT1_SV1_masssvx', 'sjetVRGT1_SV1_pb',
                    'sjetVRGT1_SV1_pc', 'sjetVRGT1_SV1_pu', 'sjetVRGT1_SV1_significance3d',
                    'sjetVRGT1_deta', 'sjetVRGT1_dphi', 'sjetVRGT1_dr', 'sjetVRGT1_eta', 'sjetVRGT1_pt',
                    'sjetVRGT1_rnnip_pb', 'sjetVRGT1_rnnip_pc', 'sjetVRGT1_rnnip_ptau', 'sjetVRGT1_rnnip_pu',
                    'sjetVRGT2_IP2D_pb', 'sjetVRGT2_IP2D_pc', 'sjetVRGT2_IP2D_pu', 'sjetVRGT2_IP3D_pb',
                    'sjetVRGT2_IP3D_pc', 'sjetVRGT2_IP3D_pu', 'sjetVRGT2_JetFitter_N2Tpair',
                    'sjetVRGT2_JetFitter_dRFlightDir', 'sjetVRGT2_JetFitter_deltaeta',
                    'sjetVRGT2_JetFitter_deltaphi', 'sjetVRGT2_JetFitter_energyFraction',
                    'sjetVRGT2_JetFitter_mass', 'sjetVRGT2_JetFitter_massUncorr',
                    'sjetVRGT2_JetFitter_nSingleTracks', 'sjetVRGT2_JetFitter_nTracksAtVtx',
                    'sjetVRGT2_JetFitter_nVTX', 'sjetVRGT2_JetFitter_significance3d',
                    'sjetVRGT2_SV1_L3d', 'sjetVRGT2_SV1_Lxy', 'sjetVRGT2_SV1_N2Tpair',
                    'sjetVRGT2_SV1_NGTinSvx', 'sjetVRGT2_SV1_deltaR', 'sjetVRGT2_SV1_dstToMatLay',
                    'sjetVRGT2_SV1_efracsvx', 'sjetVRGT2_SV1_masssvx', 'sjetVRGT2_SV1_pb', 'sjetVRGT2_SV1_pc',
                    'sjetVRGT2_SV1_pu', 'sjetVRGT2_SV1_significance3d', 'sjetVRGT2_deta', 'sjetVRGT2_dphi',
                    'sjetVRGT2_dr', 'sjetVRGT2_eta', 'sjetVRGT2_pt', 'sjetVRGT2_rnnip_pb', 'sjetVRGT2_rnnip_pc',
                    'sjetVRGT2_rnnip_ptau', 'sjetVRGT2_rnnip_pu', 'sjetVRGT3_IP2D_pb', 'sjetVRGT3_IP2D_pc',
                    'sjetVRGT3_IP2D_pu', 'sjetVRGT3_IP3D_pb', 'sjetVRGT3_IP3D_pc', 'sjetVRGT3_IP3D_pu',
                    'sjetVRGT3_JetFitter_N2Tpair', 'sjetVRGT3_JetFitter_dRFlightDir',
                    'sjetVRGT3_JetFitter_deltaeta', 'sjetVRGT3_JetFitter_deltaphi', 'sjetVRGT3_JetFitter_energyFraction',
                    'sjetVRGT3_JetFitter_mass', 'sjetVRGT3_JetFitter_massUncorr',
                    'sjetVRGT3_JetFitter_nSingleTracks', 'sjetVRGT3_JetFitter_nTracksAtVtx',
                    'sjetVRGT3_JetFitter_nVTX', 'sjetVRGT3_JetFitter_significance3d',
                    'sjetVRGT3_SV1_L3d', 'sjetVRGT3_SV1_Lxy', 'sjetVRGT3_SV1_N2Tpair', 'sjetVRGT3_SV1_NGTinSvx',
                    'sjetVRGT3_SV1_deltaR', 'sjetVRGT3_SV1_dstToMatLay', 'sjetVRGT3_SV1_efracsvx', 'sjetVRGT3_SV1_masssvx',
                    'sjetVRGT3_SV1_pb', 'sjetVRGT3_SV1_pc', 'sjetVRGT3_SV1_pu', 'sjetVRGT3_SV1_significance3d',
                    'sjetVRGT3_deta', 'sjetVRGT3_dphi', 'sjetVRGT3_dr', 'sjetVRGT3_eta', 'sjetVRGT3_pt', 'sjetVRGT3_rnnip_pb',
                    'sjetVRGT3_rnnip_pc', 'sjetVRGT3_rnnip_ptau', 'sjetVRGT3_rnnip_pu']
INPUT_VARIABLES=INPUT_VARIABLES_VR
# INPUT_VARIABLES=INPUT_VARIABLES_VRGT
INPUT_DEFAULTS={"sjetVRGT2_SV1_efracsvx": -0.44386664032936096, "sjetVRGT2_IP3D_pb": 0.024137932807207108, "sjetVRGT1_SV1_NGTinSvx": 1.7891416549682617, "sjetVRGT2_JetFitter_deltaeta": -4.478964328765869, "sjetVR1_JetFitter_deltaeta": -2.3238799571990967, "sjetVR3_rnnip_pu": 0.6309347748756409, "sjetVRGT3_rnnip_pu": 0.6309347748756409, "sjetVRGT2_IP3D_pu": 1897188.0, "fjet_ZCut12": 0.19590076804161072, "sjetVR2_JetFitter_deltaeta": -4.478964328765869, "sjetVRGT3_eta": -0.001032908447086811, "sjetVR3_SV1_masssvx": 211.4305419921875, "sjetVRGT2_SV1_NGTinSvx": 0.7536500096321106, "fjet_Split23": 19344.30078125, "sjetVR2_JetFitter_N2Tpair": 1.6975066661834717, "sjetVR1_JetFitter_N2Tpair": 3.110384941101074, "sjetVR2_JetFitter_massUncorr": 571.482666015625, "sjetVRGT2_rnnip_pc": 0.15751095116138458, "sjetVRGT1_IP3D_pc": 620627.1875, "sjetVRGT1_IP3D_pb": 0.010363494977355003, "sjetVRGT2_IP3D_pc": 1897188.0, "sjetVRGT3_SV1_Lxy": -86.33306121826172, "sjetVR1_JetFitter_dRFlightDir": -2.3079490661621094, "sjetVR2_JetFitter_mass": 1198.2474365234375, "sjetVRGT2_pt": 94500.890625, "sjetVRGT3_dr": 0.4442588686943054, "sjetVR3_JetFitter_nSingleTracks": -0.3257983326911926, "sjetVRGT3_JetFitter_energyFraction": 0.12352021783590317, "sjetVRGT3_pt": 30166.466796875, "sjetVRGT3_SV1_N2Tpair": -0.6278550028800964, "sjetVRGT1_IP3D_pu": 620627.1875, "sjetVRGT3_SV1_L3d": -85.8463363647461, "sjetVRGT2_JetFitter_N2Tpair": 1.6975066661834717, "sjetVRGT3_IP2D_pb": 0.036798834800720215, "sjetVRGT1_SV1_L3d": -33.05218505859375, "sjetVRGT3_rnnip_pc": 0.15676411986351013, "sjetVRGT3_rnnip_pb": 0.16137808561325073, "sjetVRGT1_SV1_dstToMatLay": 4.423417568206787, "sjetVRGT3_SV1_significance3d": 3.787421941757202, "fjet_mass": 138326.171875, "sjetVRGT3_JetFitter_N2Tpair": 0.0252766665071249, "sjetVR3_JetFitter_nTracksAtVtx": -0.18377499282360077, "sjetVR3_IP3D_pb": 0.0366104319691658, "sjetVR3_IP3D_pc": 5016722.0, "sjetVR2_rnnip_ptau": 0.03540557250380516, "sjetVR2_SV1_masssvx": 749.4083251953125, "sjetVRGT2_dphi": 0.0001486192486481741, "sjetVR1_SV1_efracsvx": -0.23387274146080017, "fjet_e3": 0.0019876533187925816, "sjetVR2_SV1_dstToMatLay": 3.940692186355591, "sjetVR3_SV1_Lxy": -86.33306121826172, "sjetVRGT2_SV1_deltaR": 0.017282560467720032, "fjet_Split12": 70096.4296875, "sjetVR3_dphi": -0.0003370530903339386, "fjet_KtDR": 0.5105289220809937, "sjetVR2_SV1_significance3d": 15.244283676147461, "sjetVR2_SV1_L3d": -57.831329345703125, "sjetVR2_SV1_efracsvx": -0.44386664032936096, "sjetVR1_SV1_deltaR": 0.01426777709275484, "sjetVRGT2_JetFitter_nTracksAtVtx": 1.0907983779907227, "sjetVR1_JetFitter_nVTX": 0.7090950012207031, "sjetVR2_SV1_pu": 0.64174485206604, "sjetVRGT1_JetFitter_massUncorr": 967.7078247070312, "sjetVRGT2_SV1_N2Tpair": 1.1930333375930786, "sjetVRGT2_JetFitter_mass": 1198.2474365234375, "sjetVRGT2_JetFitter_significance3d": 10.974468231201172, "sjetVR2_SV1_pb": 0.18628980219364166, "sjetVR2_SV1_pc": 0.45212188363075256, "sjetVRGT1_SV1_significance3d": 23.56785774230957, "sjetVRGT3_deta": 0.00025496038142591715, "sjetVRGT1_SV1_deltaR": 0.01426777709275484, "sjetVR3_SV1_efracsvx": -0.8024179339408875, "sjetVR2_JetFitter_nTracksAtVtx": 1.0907983779907227, "sjetVR3_JetFitter_nVTX": -0.33790498971939087, "sjetVRGT1_SV1_efracsvx": -0.23387274146080017, "sjetVR1_eta": -0.00013440585462376475, "sjetVR3_rnnip_pb": 0.16137808561325073, "sjetVRGT3_IP2D_pu": 5016722.0, "sjetVRGT1_pt": 708336.8125, "sjetVRGT1_rnnip_pu": 0.487155944108963, "sjetVRGT1_JetFitter_dRFlightDir": -2.3079490661621094, "sjetVR1_JetFitter_significance3d": 16.381288528442383, "sjetVRGT3_JetFitter_nVTX": -0.33790498971939087, "sjetVR2_SV1_N2Tpair": 1.1930333375930786, "sjetVR1_rnnip_ptau": 0.020970260724425316, "sjetVR1_SV1_L3d": -33.05218505859375, "sjetVRGT1_eta": -0.00013440585462376475, "sjetVRGT3_IP2D_pc": 5016722.0, "sjetVR1_SV1_masssvx": 1217.9443359375, "sjetVRGT1_rnnip_pc": 0.16947795450687408, "sjetVRGT1_rnnip_pb": 0.32179927825927734, "sjetVR1_SV1_Lxy": -37.608543395996094, "sjetVRGT3_SV1_masssvx": 211.4305419921875, "sjetVR2_JetFitter_nVTX": 0.36384332180023193, "sjetVR3_JetFitter_mass": 440.04388427734375, "sjetVR3_IP2D_pc": 5016722.0, "sjetVR3_IP2D_pb": 0.036798834800720215, "sjetVR2_JetFitter_dRFlightDir": -4.457581043243408, "sjetVR3_SV1_dstToMatLay": 1.2213410139083862, "sjetVR3_IP2D_pu": 5016722.0, "sjetVR2_JetFitter_significance3d": 10.974468231201172, "sjetVR1_pt": 708336.8125, "sjetVRGT3_JetFitter_nSingleTracks": -0.3257983326911926, "sjetVRGT2_rnnip_ptau": 0.03540557250380516, "sjetVR3_SV1_N2Tpair": -0.6278550028800964, "sjetVRGT2_SV1_masssvx": 749.4083251953125, "sjetVRGT1_JetFitter_mass": 2198.052490234375, "sjetVRGT3_SV1_efracsvx": -0.8024179339408875, "sjetVR3_JetFitter_energyFraction": 0.12352021783590317, "sjetVRGT2_JetFitter_nVTX": 0.36384332180023193, "sjetVRGT1_deta": -4.482616077439161e-06, "sjetVRGT2_SV1_pc": 0.45212188363075256, "sjetVRGT2_SV1_pb": 0.18628980219364166, "sjetVRGT3_JetFitter_deltaphi": -7.042611122131348, "sjetVR1_SV1_dstToMatLay": 4.423417568206787, "fjet_D2": 2.374298572540283, "sjetVRGT2_SV1_pu": 0.64174485206604, "sjetVRGT1_rnnip_ptau": 0.020970260724425316, "sjetVRGT2_SV1_significance3d": 15.244283676147461, "sjetVRGT3_rnnip_ptau": 0.045890919864177704, "sjetVRGT2_SV1_dstToMatLay": 3.940692186355591, "sjetVRGT2_SV1_Lxy": -59.74996566772461, "sjetVR2_rnnip_pu": 0.5210351347923279, "sjetVR1_deta": -4.482616077439161e-06, "sjetVRGT2_JetFitter_dRFlightDir": -4.457581043243408, "sjetVRGT2_SV1_L3d": -57.831329345703125, "sjetVRGT2_dr": 0.3145780861377716, "sjetVR3_JetFitter_dRFlightDir": -7.021878242492676, "fjet_Aplanarity": -6634.8115234375, "fjet_eta": -2.7349522497388534e-05, "sjetVR1_IP3D_pu": 620627.1875, "sjetVRGT2_JetFitter_deltaphi": -4.479055404663086, "sjetVR2_rnnip_pc": 0.15751095116138458, "sjetVR2_rnnip_pb": 0.28414133191108704, "sjetVRGT3_SV1_NGTinSvx": -0.6653450131416321, "sjetVR3_eta": -0.001032908447086811, "sjetVRGT2_JetFitter_massUncorr": 571.482666015625, "sjetVR1_JetFitter_mass": 2198.052490234375, "sjetVRGT2_JetFitter_energyFraction": 0.268054723739624, "sjetVRGT1_JetFitter_nVTX": 0.7090950012207031, "fjet_Tau32_wta": 0.5257828831672668, "sjetVRGT3_JetFitter_deltaeta": -7.042212009429932, "sjetVR1_IP3D_pb": 0.010363494977355003, "sjetVR1_IP3D_pc": 620627.1875, "sjetVRGT1_SV1_N2Tpair": 2.4951915740966797, "sjetVR3_deta": 0.00025496038142591715, "sjetVRGT1_JetFitter_deltaphi": -2.323951244354248, "sjetVR3_rnnip_pc": 0.15676411986351013, "sjetVR1_SV1_pu": 0.5038563013076782, "sjetVRGT3_SV1_pu": 0.8573762774467468, "sjetVR3_SV1_pu": 0.8573762774467468, "fjet_PlanarFlow": 0.4177548587322235, "sjetVRGT3_IP3D_pu": 5016722.0, "sjetVR3_dr": 0.4442588686943054, "sjetVRGT1_IP2D_pu": 620627.1875, "sjetVR3_SV1_pc": 0.5991700291633606, "sjetVR2_dphi": 0.0001486192486481741, "sjetVR1_SV1_pc": 0.35531526803970337, "sjetVR1_SV1_pb": 0.17023292183876038, "sjetVRGT2_deta": -0.0003230147995054722, "sjetVR2_IP3D_pc": 1897188.0, "sjetVR2_IP3D_pb": 0.024137932807207108, "sjetVRGT1_IP2D_pb": 0.010208160616457462, "sjetVRGT1_IP2D_pc": 620627.1875, "sjetVRGT3_IP3D_pc": 5016722.0, "sjetVRGT3_IP3D_pb": 0.0366104319691658, "sjetVR2_SV1_NGTinSvx": 0.7536500096321106, "sjetVR3_JetFitter_deltaphi": -7.042611122131348, "sjetVR3_JetFitter_massUncorr": 200.05670166015625, "sjetVRGT1_JetFitter_nSingleTracks": 0.6342599987983704, "sjetVR1_SV1_significance3d": 23.56785774230957, "sjetVR3_SV1_NGTinSvx": -0.6653450131416321, "sjetVR1_SV1_N2Tpair": 2.4951915740966797, "sjetVR3_SV1_pb": 0.2110213190317154, "fjet_Tau21_wta": 0.38430055975914, "sjetVR2_dr": 0.3145780861377716, "sjetVRGT2_eta": -0.00018497538985684514, "sjetVR1_JetFitter_deltaphi": -2.323951244354248, "sjetVRGT3_SV1_deltaR": 0.01059037446975708, "sjetVR2_IP3D_pu": 1897188.0, "sjetVR1_rnnip_pu": 0.487155944108963, "sjetVR1_IP2D_pu": 620627.1875, "sjetVR3_IP3D_pu": 5016722.0, "sjetVR2_deta": -0.0003230147995054722, "sjetVR3_rnnip_ptau": 0.045890919864177704, "sjetVR1_JetFitter_energyFraction": 0.373908132314682, "sjetVR3_JetFitter_deltaeta": -7.042212009429932, "sjetVR1_IP2D_pc": 620627.1875, "sjetVR1_IP2D_pb": 0.010208160616457462, "sjetVR1_rnnip_pb": 0.32179927825927734, "sjetVR1_rnnip_pc": 0.16947795450687408, "fjet_pt": 1119039.375, "sjetVRGT1_JetFitter_energyFraction": 0.373908132314682, "sjetVRGT1_dphi": 0.00025019902386702597, "sjetVRGT3_JetFitter_significance3d": 3.560934543609619, "sjetVR3_JetFitter_N2Tpair": 0.0252766665071249, "sjetVR3_SV1_significance3d": 3.787421941757202, "fjet_C2": 0.11352093517780304, "sjetVRGT1_SV1_Lxy": -37.608543395996094, "sjetVRGT2_rnnip_pu": 0.5210351347923279, "sjetVR1_dr": 0.14340455830097198, "sjetVRGT1_SV1_pb": 0.17023292183876038, "sjetVRGT1_SV1_pc": 0.35531526803970337, "sjetVRGT2_rnnip_pb": 0.28414133191108704, "sjetVRGT1_JetFitter_deltaeta": -2.3238799571990967, "sjetVRGT1_JetFitter_significance3d": 16.381288528442383, "sjetVR2_JetFitter_nSingleTracks": 0.29755499958992004, "sjetVR3_JetFitter_significance3d": 3.560934543609619, "fjet_Qw": 39145.64453125, "sjetVRGT1_SV1_pu": 0.5038563013076782, "sjetVR2_SV1_deltaR": 0.017282560467720032, "sjetVR2_JetFitter_energyFraction": 0.268054723739624, "fjet_FoxWolfram20": 0.45841842889785767, "sjetVRGT1_JetFitter_nTracksAtVtx": 1.8320599794387817, "sjetVRGT3_SV1_dstToMatLay": 1.2213410139083862, "sjetVR3_SV1_L3d": -85.8463363647461, "sjetVR1_JetFitter_nSingleTracks": 0.6342599987983704, "sjetVR1_JetFitter_massUncorr": 967.7078247070312, "sjetVR2_eta": -0.00018497538985684514, "sjetVR1_SV1_NGTinSvx": 1.7891416549682617, "sjetVRGT1_JetFitter_N2Tpair": 3.110384941101074, "sjetVRGT1_dr": 0.14340455830097198, "sjetVR2_SV1_Lxy": -59.74996566772461, "sjetVRGT2_JetFitter_nSingleTracks": 0.29755499958992004, "sjetVR2_pt": 94500.890625, "sjetVRGT3_JetFitter_mass": 440.04388427734375, "sjetVR3_pt": 30166.466796875, "sjetVRGT3_JetFitter_dRFlightDir": -7.021878242492676, "fjet_Angularity": -0.002602761145681143, "sjetVR2_JetFitter_deltaphi": -4.479055404663086, "sjetVR2_IP2D_pb": 0.023645393550395966, "sjetVR2_IP2D_pc": 1897188.0, "sjetVRGT3_dphi": -0.0003370530903339386, "sjetVRGT3_SV1_pb": 0.2110213190317154, "sjetVRGT2_IP2D_pc": 1897188.0, "sjetVRGT2_IP2D_pb": 0.023645393550395966, "sjetVRGT3_JetFitter_nTracksAtVtx": -0.18377499282360077, "sjetVR2_IP2D_pu": 1897188.0, "sjetVRGT3_SV1_pc": 0.5991700291633606, "sjetVRGT3_JetFitter_massUncorr": 200.05670166015625, "sjetVR1_JetFitter_nTracksAtVtx": 1.8320599794387817, "sjetVR3_SV1_deltaR": 0.01059037446975708, "sjetVRGT2_IP2D_pu": 1897188.0, "sjetVRGT1_SV1_masssvx": 1217.9443359375, "sjetVR1_dphi": 0.00025019902386702597}
USED_VARIABLES=list(set(INPUT_VARIABLES+DECORRELATION_VARIABLES+
                        DECORRELATION_VARIABLES_AUX+
                        WEIGHT_VARIABLES+FLAG_VARIABLES))

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
def load_data (path, name='dataset', train=None, test=None, signal=None, background=None, sample=None, seed=21, replace=True,fillna=True,dropna=False):
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
    # data = pd.read_hdf(path, name)[USED_VARIABLES]
    data = pd.read_hdf(path, name)
    log.info("examine input data\n{}".format(
        data.info()
    ))
    log.debug("examine input data without filling\n{}".format(
        data.describe()
    ))
    log.info("N/A report:")
    log.info(data.isna().sum())
    if fillna:
        log.info("N/A filling with defaults...")
        data=data.fillna(value=INPUT_DEFAULTS)
        log.debug("examine input data after filling\n{}".format(
            data.describe()
        ))
        log.info("N/A report NOW:")
        log.info(data.isna().sum())
    elif dropna:
        log.info("N/A dropped in any colomn...")
        data = data.dropna()
        log.debug("examine input data after dropping\n{}".format(
            data.describe()
        ))
        log.info("N/A report NOW:")
        log.info(data.isna().sum())

    # Subsample signal by x10 for testing: 1E+07 -> 1E+06?????????
    # np.random.seed(7)
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
    # print data.dtypes
    return data, features_input, features_decorrelation

# @garbage_collect
# @profile
# def load_data_raw (path, name='dataset', train=None, test=None, signal=None, background=None, sample=None, seed=21, replace=True):
#     """
#     General script to load data, common to all run scripts. Without fill NaN
#
#     Arguments:
#         path: The path to the HDF5 file, from which data should be loaded.
#         name: Name of the dataset, as stored in the HDF5 file.
#         ...
#
#     Returns:
#         Tuple of pandas.DataFrame containing the loaded; list of loaded features
#         to be used for training; and list of features to be used for mass-
#         decorrelation.
#
#     Raises:
#         IOError: If no HDF5 file exists at the specified `path`.
#         KeyError: If the HDF5 does not contained a dataset named `name`.
#         KeyError: If any of the necessary features are not present in the loaded
#             dataset.
#     """
#
#     # Check(s)
#     assert False not in [train, test, signal, background]
#     if sample: assert 0 < sample and sample < 1.
#
#     # Read data from HDF5 file
#     data = pd.read_hdf(path, name)
#     print "load_data: N/A report: "
#     print data.isna().sum()
#     print "load_data: N/A filled with defaults"
#     print "examine without fill: "
#     print data.describe()
#
#     # Subsample signal by x10 for testing: 1E+07 -> 1E+06?????????
#     np.random.seed(7)
#     try:
#         msk_test  = data['train'].astype(bool)
#         msk_train = ~msk_test #train/test disting
#         msk_bkg = data['signal'].astype(bool)
#         msk_sig = ~msk_bkg  #signal/bkg
#
#         # idx_sig = np.where(msk_sig)[0]
#         # idx_sig = np.random.choice(idx_sig, int(msk_sig.sum() * 0.1), replace=False) #select 10%
#         # msk_sig = np.zeros_like(msk_bkg).astype(bool)
#         # msk_sig[idx_sig] = True #reset selected sig
#         # data = data[msk_train | (msk_test & (msk_sig | msk_bkg))]
#         pass
#     except:
#         log.warning("Some of the keys ['train', 'signal'] were not present in file {}".format(path))
#         pass
#
#     # Logging
#     try:
#         for sig, name in zip([True, False], ['signal', 'background']):
#             log.info("Found {:8.0f} training and {:8.0f} test samples for {}".format(
#                 sum((data['signal'] == sig) & (data['train'] == True)),
#                 sum((data['signal'] == sig) & (data['train'] == False)),
#                 name
#                 ))
#             pass
#     except KeyError:
#         log.info("Some key(s) in data were not found")
#         pass
#
#     # Define feature collections to use
#     features_input         = INPUT_VARIABLES
#     features_decorrelation = DECORRELATION_VARIABLES
#
#     # Split data, for different usage
#     if train:
#         log.info("load_data: Selecting only training data.")
#         data = data[data['train']  == True]
#         pass
#
#     if test:
#         log.info("load_data: Selecting only testing data.")
#         data = data[data['train']  == False]
#         pass
#
#     if signal:
#         log.info("load_data: Selecting only signal data.")
#         data = data[data['signal'] == True]
#         pass
#
#     if background:
#         log.info("load_data: Selecting only background data.")
#         data = data[data['signal'] == False]
#         pass
#
#     if sample:
#         log.info("load_data: Selecting a random fraction {:.2f} of data (replace = {}, seed = {}).".format(sample, replace, seed))
#         data = data.sample(frac=sample, random_state=seed, replace=False) #dataframe.sample
#         # no replace means one element can only be selected once
#         pass
#     # Return
#     print data.dtypes
#     return data, features_input, features_decorrelation
