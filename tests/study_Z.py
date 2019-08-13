#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing study partZ -- examine the input (training/test samples)"""

# Basic import(s)
# import re
# import gc
# import gzip
import itertools
from prepro.common import run_batched

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Scientific import(s)
# import numpy as np
# import pandas as pd
# import pickle
# import root_numpy
# from array import array
# from scipy.stats import entropy
# from sklearn.metrics import roc_curve, roc_auc_score

# Project import(s)
from adversarial.utils import initialise, initialise_backend, parse_args, load_data,load_data_raw, mkdir, wpercentile, latex
from adversarial.profile import profile, Profile
# from adversarial.constants import *
from run.adversarial.common import initialise_config
from .studies.common import *
import studies.samplesexam
# import os

# Custom import(s)
# import rootplotting as rp

#now all keep 4 test and addition samplesexame(), all without weight!
# Main function definition
@profile
def main (args):

    # Initialise
    args, _ = initialise(args)
    # Loading
    DATA, _, _ = load_data(args.input + 'data.h5',fillna=False) #now test mem share across thread
    # Examining
    exam_samples(DATA, args, testOnly=False)
    return 0

def exam_samples(data, args, testOnly=False,features=None):
        """
        Method exam samples.
        """
        #masscuts = [True, False]
        trains=[None] if testOnly else [None,True,False]
        pt_ranges = [None, (200 * GeV, 500 * GeV), (500 * GeV, 1000 * GeV), (1000 * GeV, 2000 * GeV)]
        if args.debug: pt_ranges = [None] #debug
        if features is None:
            # features=set(INPUT_VARIABLES,DECORRELATION_VARIABLES,
            #              WEIGHT_VARIABLES,DECORRELATION_VARIABLES_AUX)
            features=list(data)
        # data=data[features].dropna()
        # data = data.dropna() # Not drop Na here!
        with Profile("Study: samples variables"):
            mass_ranges = np.linspace(50*GeV, 300*GeV, 5 + 1, endpoint=True)
            mass_ranges = [None,(80*GeV,140*GeV)] + zip(mass_ranges[:-1], mass_ranges[1:])
            if args.debug: mass_ranges =[None] #debug
            fillnas=[False,True]
            # for feat, pt_range, mass_range,train,fillna in itertools.product(features, pt_ranges,
            #                                                     mass_ranges,trains,fillnas):
            #     # studies.samplesexam(data, args, feat, pt_range, mass_range,train,fillna)
            #     pass
            # argsBatch=itertools.product([data],[args],features, pt_ranges,mass_ranges,trains,fillnas)
            argsBatch1=itertools.product([args],features, pt_ranges,mass_ranges,trains,fillnas)
            argsBatch2=list(argsBatch1)
            argsBatch3=[[dataSlice(data,ARG)]+[ARG] for ARG in argsBatch2]
            # if args.debug: print "batchArgs",len(list(argsBatch))
            run_batched(studies.samplesChecker,argsBatch3,args.max)
            pass

# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass

def dataSlice(data,ARG):
    _, feature, _, _, _, _=ARG
    # partList=list(set([feature]+[PT,MASS]+["signal","train"]+WEIGHT_VARIABLES))
    partList=list(set([feature]+[PT,MASS]+["signal","train"])
    dataPart=data[partList]
    return dataPart