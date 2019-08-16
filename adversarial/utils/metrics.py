 # -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_curve
from .setup import DECORRELATION_VARIABLES, WEIGHT_VARIABLES, DECORRELATION_VARIABLES_AUX,INPUT_VARIABLES
PT=DECORRELATION_VARIABLES_AUX[0]
MASS=DECORRELATION_VARIABLES[0]
GeV=1000.0


# Project imports
from adversarial.utils import garbage_collect

# Global variable definition(s)
MASSBINS = np.linspace(50*GeV, 300*GeV, (300 - 50) // 5 + 1, endpoint=True)
MASSRANGE=[50.*GeV,300.*GeV]

def signal_low (feat):
    """Method to determine whether the signal distribution is towards higher values."""
    return ('Tau21' in feat or 'D2' in feat or 'N2' in feat)


def JSD (P, Q, base=2):
    """Compute Jensen-Shannon divergence (JSD) of two distribtions.
    From: [https://stackoverflow.com/a/27432724]

    Arguments:
        P: First distribution of variable as a numpy array.
        Q: Second distribution of variable as a numpy array.
        base: Logarithmic base to use when computing KL-divergence.

    Returns:
        Jensen-Shannon divergence of `P` and `Q`.
    """
    if np.sum(P)==0 or np.sum(Q)==0 :
        print "Error! JSD input is all 0"
    p = P / np.sum(P)
    q = Q / np.sum(Q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=base) + entropy(q, m, base=base))


@garbage_collect
def metrics (data, feat, target_tpr=0.5, cut=None, masscut=False, verbose=False):
    """
    Compute the standard metrics (bkg. rejection and JSD) from a DataFrame.
    Assuming that any necessary selection has already been imposed.

    Arguments:
        data: Pandas dataframe, assumed to hold all necessary columns (signal,
            weight_test, m, and `feat`) and to have been subjected to the
            nessecary selection up-stream (training/test split, phase space
            restriction.)
        feat: Name of feateure for which to compute the standard metrics.
        target_tpr: Signal efficiency at which to compute the standard metrics.
        masscut: Whether to impose additional 60 GeV < m < 100 GeV cut.
        verbose: Whether to print information to stdout.

    Returns:
        Tuple of (background rejection at `target_tpr`, JSD for background mass
        distributions at `target_tpr`).
    """
    print "metrics calu started..."
    # Background rejection at `target_tpr` signal efficiency
    # ------------------------------------------------------

    # (Opt.) mass cut mask
    if masscut:
        print "metrics: Applying mass cut."
        # msk = (data[M] > 60.*GeV) & (data[M] < 100.*GeV) if masscut else np.ones_like(data['signal']).astype(bool)
        pass

    msk = (data[MASS] > 80. * GeV) & (data[MASS] < 140. * GeV) if masscut else np.ones_like(data['signal']).astype(bool) # mass cut From Wei


    # scikit-learn assumes signal towards 1, background towards 0
    pred = data[feat].values.copy()
    if signal_low(feat):
        if verbose:
            print "metrics: Reversing cut direction for {}".format(feat)
            pass
        pred *= -1.
        pass

    # Compute ROC curve efficiencies
    # fpr, tpr, thresholds = roc_curve(data.loc[msk, 'signal'], pred[msk], sample_weight=data.loc[msk, 'weight_test'])
    #print type(data.loc[msk, 'signal']),data.loc[msk, 'signal'].sum(),data.loc[msk, 'signal'].size
    #print type(data.loc[msk, 'signal'].values), data.loc[msk, 'signal'].values.sum(), data.loc[msk, 'signal'].values.size
    #print type(pred[msk]),pred[msk].sum(),pred[msk].size
    print "msk",msk.sum(),msk.size
    print "data",data.loc[msk, 'signal'].values.sum(),"pred",pred[msk].sum()
    fpr, tpr, thresholds = roc_curve(data.loc[msk, 'signal'].values, pred[msk])
    #print "fpr",fpr,"tpr",tpr,"thre",thresholds
    if masscut:
        tpr_mass = np.mean(msk[data['signal'] == 1])
        fpr_mass = np.mean(msk[data['signal'] == 0])

        print "tpr_mass",tpr_mass
        print "fpr_mass",fpr_mass
        tpr *= tpr_mass
        fpr *= fpr_mass
        pass

    # Get background rejection factor
    if cut is None:
        idx = np.argmin(np.abs(tpr - target_tpr))
        cut = thresholds[idx]    
        print "No manual cut: ","idx",idx,"cut",cut,"tpr,fpr@idx",tpr[idx],fpr[idx]
    else:
        print "metrics: Using manual cut of {:.2f} for {}".format(cut, feat)
        idx = np.argmin(np.abs(thresholds - cut))
        print "idx",idx,"tpr,fpr@idx",tpr[idx],fpr[idx]
    print "metrics:   effsig = {:.1f}%, effbkg = {:.1f}, threshold = {:.2f}".format(tpr[idx] * 100.,
                                                                                        fpr[idx] * 100.,
                                                                                        thresholds[idx])

    # @FIXME
    # here sometimes the fpr[idx] will become 0, don't know why.when masscut=True and cut=None

    eff = tpr[idx]
    if fpr[idx]==0. :
        print "fpr@{} where tpr==target_tpr is 0!!".format(idx)
        print "fpr==0 counts", np.count_nonzero(fpr==0),fpr.size
        # rej=np.nan_to_num(np.inf)
    rej = 1. / fpr[idx]


    # JSD at `target_tpr` signal efficiency
    # -------------------------------------

    # Get JSD(1/Npass dNpass/dm || 1/Nfail dNfail/dm)
    msk_pass = pred > cut
    msk_bkg  = data['signal'] == 0
    print "msk_pass",msk_pass.sum(),msk_pass.size
    print "msk_bkg",msk_bkg.sum(),msk_bkg.size

    # p, _ = np.histogram(data.loc[ msk_pass & msk_bkg, M].values, bins=MASSBINS, weights=data.loc[ msk_pass & msk_bkg, 'weight_test'].values, density=1.)
    # f, _ = np.histogram(data.loc[~msk_pass & msk_bkg, M].values, bins=MASSBINS, weights=data.loc[~msk_pass & msk_bkg, 'weight_test'].values, density=1.)
    print "p data",data.loc[msk_pass & msk_bkg, MASS].values.sum(),data.loc[msk_pass & msk_bkg, MASS].values.size
    print "f data",data.loc[~msk_pass & msk_bkg, MASS].values.sum(),data.loc[~msk_pass & msk_bkg, MASS].values.size
    p, _ = np.histogram(data.loc[msk_pass & msk_bkg, MASS].values, bins=MASSBINS, density=1.)
    f, _ = np.histogram(data.loc[~msk_pass & msk_bkg, MASS].values, bins=MASSBINS, density=1.)
    print "p", p.sum(), p.size
    print "f", f.sum(), f.size

    jsd = JSD(p, f)

    # Return metrics
    print "eff",eff,"rej", rej,"jsd", jsd
    return eff, rej, 1./jsd #??


@garbage_collect
def bootstrap_metrics (data, feat, num_bootstrap=10, **kwargs):
    """
    ...
    """
    # Compute metrics using bootstrapping
    print "booststrap_metrics calu started...",feat
    bootstrap_eff, bootstrap_rej, bootstrap_jsd = list(), list(), list()
    for i in range(num_bootstrap):
        print "round",i+1,"/",num_bootstrap
        idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
        eff, rej, jsd = metrics(data.iloc[idx], feat, **kwargs)
        if not np.isfinite([eff,rej,jsd]).sum()==3:
            print "Error in this round, skip!! pls check yourself"
            continue
        bootstrap_eff.append(eff)
        bootstrap_rej.append(rej)
        bootstrap_jsd.append(jsd)
        pass
    print "booststrap_metrics ok!",len(bootstrap_eff),len(bootstrap_rej),len(bootstrap_jsd)
    return (np.mean(bootstrap_eff), np.std(bootstrap_eff)), \
           (np.mean(bootstrap_rej), np.std(bootstrap_rej)), \
           (np.mean(bootstrap_jsd), np.std(bootstrap_jsd))
