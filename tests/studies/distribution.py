#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
import ROOT

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile, garbage_collect
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp

#Now without weight_test


# Global variable definition(s)
HISTSTYLE[True] ['label'] = "#it{Hbb} jets"
HISTSTYLE[False]['label'] = "Bkg"


@garbage_collect #not enough due to python's limit. might use multi-thread to release the mem.
@showsave
def distribution (data_, args, feat, pt_range, mass_range):
    """
    Perform study of substructure variable distributions.

    Saves plot `figures/distribution_[feat].pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        feat: Feature for which to plot signal- and background distributions.
    """

    # Select data
    if pt_range is not None:
        data = data_[(data_[PT] > pt_range[0]) & (data_[PT] < pt_range[1])]
    else:
        data = data_
        pass

    if mass_range is not None:
        data = data[(data[MASS] > mass_range[0]) & (data[MASS] < mass_range[1])]
        pass

    # Define bins
    # xmin = wpercentile (data[feat].values,  1, weights=data['weight_test'].values)
    # xmax = wpercentile (data[feat].values, 99, weights=data['weight_test'].values)
    xmin = wpercentile (data[feat].values,  1)
    xmax = wpercentile (data[feat].values, 99)

    # if   feat == 'D2-k#minusNN':
    #     print "distribution: kNN feature '{}'".format(feat)
    #     xmin, xmax = -1.,  2.
    # elif feat.lower().startswith('d2'):
    #     print "distribution: D2  feature '{}'".format(feat)
    #     xmin, xmax =  0.,  3.
    # elif 'tau21' in feat.lower():
    #     xmin, xmax =  0.,  1.
    #     pass

    snap = 0.5  # Snap to nearest multiple in appropriate direction
    xmin = np.floor(xmin / snap) * snap
    xmax = np.ceil (xmax / snap) * snap

    bins = np.linspace(xmin, xmax, 50 + 1, endpoint=True)

    # Perform plotting
    c = plot(args, data, feat, bins, pt_range, mass_range)

    # Output
    path = 'figures/distribution/distribution_{}{}{}'.format(standardise(feat), '__pT{:.0f}_{:.0f}'.format(pt_range[0]/GeV, pt_range[1]/GeV) if pt_range is not None else '', '__mass{:.0f}_{:.0f}'.format(mass_range[0]/GeV, mass_range[1]/GeV) if mass_range is not None else '')
    path = path+args.bkg+".pdf"

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    args, data, feat, bins, pt_range, mass_range = argv

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Style
    histstyle = dict(**HISTSTYLE)
    base = dict(bins=bins, alpha=0.5, normalise=True, linewidth=3,label=args.bkg)

    # Plots
    for signal in [False,True]: #Note: HDF's native bool type is not easy. so bool when write to hdf will become to int8. 8x waste, but safe.
        msk = (data['signal'] == int(signal))
        histstyle[signal].update(base)
        # c.hist(data.loc[msk, feat].values, weights=data.loc[msk, 'weight_test'].values, **histstyle[signal])
        c.hist(data.loc[msk, feat].values, **histstyle[signal])
        pass

    # Decorations
    c.xlabel("Large-#it{R} jet " + latex(feat, ROOT=True))
    c.ylabel("Fraction of jets")
    if args.bkg == "D":
        bkg="Hbb v.s. Dijets(subsampling)"
    elif args.bkg=="T":
        bkg = "Hbb v.s. Top"
    else:
        bkg = ""
        pass
    c.text(TEXT + ["dataset p3652","#it{Hbb} tagging",bkg] + (
        ["p_{{T}} #in  [{:.0f}, {:.0f}] GeV".format(pt_range[0], pt_range[1])] if pt_range is not None else []
        ) + (
        ["m #in  [{:.0f}, {:.0f}] GeV".format(mass_range[0], mass_range[1]),] if mass_range is not None else []
        ), qualifier=QUALIFIER)
    c.ylim(4E-03, 4E-01)
    c.logy()
    c.legend()
    return c
