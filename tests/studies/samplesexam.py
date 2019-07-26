#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
# import ROOT

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile, garbage_collect
from adversarial.constants import *
from adversarial.utils.setup import INPUT_DEFAULTS
import multiprocessing

# Custom import(s)
import rootplotting as rp

# Now without weight_test

class samplesChecker (multiprocessing.Process):

    # Global variable definition(s)
    HISTSTYLE[True]['label'] = "#it{Hbb} "
    HISTSTYLE[False]['label'] = "Dijets"

    def __init__ (self, vargs):
        """
        Process for examing samples.

        Arguments:
            path: Path to the ROOT file to be converted.
            key: Class to which the file pointer to by `path` belongs
            args: Namespace containing command-line arguments, to configure the
                reading and writing of files.
        """

        # Unpack input arguments
        # data_, args, feat, *aux = vargs #not work on python2!!!
        #pt_range = None, mass_range = None, train = None, fillna = True
        self.__vargs=vargs

        # Base class constructor
        super(samplesChecker, self).__init__()

        # Member variable(s)
        # self.__data = data_
        # self.__args = args
        # self.__feat = feat
        # self.__pt_range = aux[0] if aux else None
        # self.__mass_range = aux[1] if len(aux)>1 else None
        # self.__train = aux[2] if len(aux)>2 else None
        # self.__fillna = aux[3] if len(aux)>3 else True
        return
    def run (self):
        print "Start process"
        # self.samplesexam(self.__vargs[0],self.__vargs[1],self.__vargs[2],self.__vargs[3] if len(self.__vargs)>3 else [])
        print "vArgs",self.__vargs
        self.samplesexam(*self.__vargs)
        # self.samplesexam(self.__data,self.__args,self.__feat,self.__pt_range,self.__mass_range,self.__train,self.__fillna)
        #now test sample wrapper, if possible, complete class it!!
        print "End process"
        return

    @garbage_collect
    @showsave
    def samplesexam(self,data_, args, feat, pt_range = None, mass_range = None, train = None, fillna = True):
        """
        Perform study of substructure variable distributions.

        Saves plot `figures/distribution_[feat].pdf`

        Arguments:
            data_: Pandas data frame from which to read data.
            args: Namespace holding command-line arguments.
            feat: Feature for which to plot.
        """
        if args.debug:print "Start child, Arg",data_.size, args, feat, pt_range, mass_range, train, fillna
        if train is not None:
            if train:
                data = data_[data_['train'] == True]  # note train =1/0, compare with True/False is ok.
                print "select train", feat, data.size
                pass
            else:
                data = data_[data_['train'] == False]
                print "select test", feat, data.size
                pass
        else:
            data = data_
            print "select all", feat, data.size
            pass
        assert data.is_copy
        # Select data
        if pt_range is not None:
            data = data[(data[PT] > pt_range[0]) & (data[PT] < pt_range[1])]

        if mass_range is not None:
            data = data[(data[M] > mass_range[0]) & (data[M] < mass_range[1])]

        if fillna:
            if data[feat].isna().sum() == 0:
                print "No need fillna,exit.", feat
                return
            else:
                print "fill N/A", feat, data[feat].isna().sum()
                # data[feat].fillna(value=INPUT_DEFAULTS,inplace=True) #inplace is annoying in python...pay attention to copy!!
                data = data.fillna(value=INPUT_DEFAULTS)

        # Define bins
        # xmin = wpercentile (data[feat].values,  1, weights=data['weight_test'].values)
        # xmax = wpercentile (data[feat].values, 99, weights=data['weight_test'].values)
        print ">>sample exam", feat, pt_range, mass_range, train, data[feat].size
        xmin = wpercentile(data[feat].values, 1)
        xmax = wpercentile(data[feat].values, 99)
        if args.debug: print "xmin,xmax", xmin, xmax
        if not xmax > xmin:  # wrong values or nan
            print "Error, xmax xmin is not valid, exit.", feat
            return  # return empty so that no saving

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
        xmax = np.ceil(xmax / snap) * snap

        bins = np.linspace(xmin, xmax, 50 + 1, endpoint=True)

        # Perform plotting
        c = self.plot(args, data, feat, bins, pt_range, mass_range)

        # Output
        outPath = args.output.rstrip("/")
        path = outPath + '/samples/{}-{}_{}{}{}.pdf'.format(standardise(feat),
                                                            "all" if train is None else ("train" if train else "test"),
                                                            '__pT{:.0f}_{:.0f}'.format(pt_range[0], pt_range[
                                                                1]) if pt_range is not None else '',
                                                            '__mass{:.0f}_{:.0f}'.format(mass_range[0], mass_range[
                                                                1]) if mass_range is not None else '',
                                                            "_raw" if not fillna else "")
        if args.debug: print path
        return c, args, path

    # @staticmethod
    def plot(self,*argv):
        """
        Method for delegating plotting.
        """

        # Unpack arguments
        args, data, feat, bins, pt_range, mass_range = argv

        # Canvas
        c = rp.canvas(batch=not args.show)

        # Style
        histstyle = dict(**HISTSTYLE)
        base = dict(bins=bins, alpha=0.5, normalise=True, linewidth=3)

        # Plots
        for signal in [False, True]:
            msk = (data['signal'] == signal)
            histstyle[signal].update(base)
            # c.hist(data.loc[msk, feat].values, weights=data.loc[msk, 'weight_test'].values, **histstyle[signal])
            c.hist(data.loc[msk, feat].values, **histstyle[signal])
            pass

        # Decorations
        c.xlabel("Large-#it{R} jet " + latex(feat, ROOT=True))
        c.ylabel("Fraction of jets")
        c.text(TEXT + [
            "#it{Hbb} tagging"] + (
                   ["p_{{T}} #in  [{:.0f}, {:.0f}] GeV".format(pt_range[0],
                                                               pt_range[1])] if pt_range is not None else []
               ) + (
                   ["m #in  [{:.0f}, {:.0f}] GeV".format(mass_range[0],
                                                         mass_range[1]), ] if mass_range is not None else []
               ), qualifier=QUALIFIER)
        c.ylim(4E-03, 4E-01)
        c.logy()
        c.legend()
        return c
    pass





