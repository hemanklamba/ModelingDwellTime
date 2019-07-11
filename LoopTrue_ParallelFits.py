import os
import sys
import numpy as np
import pandas as pd
import google.auth
import time
from datetime import timedelta
import dask.dataframe as dd
from datetime import datetime
from scipy import stats
import matplotlib.colors as colors
from scipy.stats import rankdata
from progressbar import ProgressBar
from fitter import Fitter
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import fisk, invgauss, lognorm, expon, weibull_min, genpareto, powerlaw, gamma
from scipy.stats import ks_2samp

import cPickle as pickle
from joblib import Parallel, delayed
import warnings
import sys

def compute_BIC(data, n_params, neg_ll):
    bic = (np.log(data.shape[0]) * n_params) - 2 * -neg_ll
    return bic

def run_Parametric(data):
    print "Fitting Fisk"
    fisk_params = fisk.fit(data, floc=0)
    fisk_nll = fisk.nnlf(fisk_params, data)
    fisk_rvs = fisk.rvs(*fisk_params, size=data.shape[0])
    ks_fisk = ks_2samp(data, fisk_rvs)
    bic_fisk = compute_BIC(data, len(fisk_params), fisk_nll)

    print "Fitting IG"
    ig_params = invgauss.fit(data, floc=0)
    ig_nll = invgauss.nnlf(ig_params, data)
    ig_rvs = invgauss.rvs(*ig_params, size=data.shape[0])
    ks_ig = ks_2samp(data, ig_rvs)
    bic_ig = compute_BIC(data, len(ig_params), ig_nll)

    print "Fitting LN"
    ln_params = lognorm.fit(data, floc=0)
    ln_nll = lognorm.nnlf(ln_params, data)
    ln_rvs = lognorm.rvs(*ln_params, size=data.shape[0])
    ks_ln = ks_2samp(data, ln_rvs)
    bic_ln = compute_BIC(data, len(ln_params), ln_nll)

    #print "Fitting GP"
    #gp_params = genpareto.fit(data, floc=0)
    #gp_nll = genpareto.nnlf(gp_params, data)
    #gp_rvs = genpareto.rvs(*gp_params, size=data.shape[0])
    #ks_gp = ks_2samp(data, gp_rvs)

    #print "Fitting PL"
    #pl_params = powerlaw.fit(data, floc=0)
    #pl_nll = powerlaw.nnlf(pl_params, data)
    #pl_rvs = powerlaw.rvs(*pl_params, size=data.shape[0])
    #ks_pl = ks_2samp(data, pl_rvs)

    #print "Fitting EXP"
    #exp_params = expon.fit(data, floc=0)
    #exp_nll = expon.nnlf(exp_params, data)
    #exp_rvs = expon.rvs(*exp_params, size=data.shape[0])
    #ks_exp = ks_2samp(data, exp_rvs)

    print "Fitting Weibull"
    weib_params = weibull_min.fit(data, floc=0)
    weib_nll = weibull_min.nnlf(weib_params, data)
    weib_rvs = weibull_min.rvs(*weib_params, size=data.shape[0])
    ks_weib = ks_2samp(data, weib_rvs)
    bic_weib = compute_BIC(data, len(weib_params), weib_nll)

    print "Fitting Gamma"
    gamma_params = gamma.fit(data, floc=0)
    gamma_nll = gamma.nnlf(gamma_params, data)
    gamma_rvs = gamma.rvs(*gamma_params, size=data.shape[0])
    ks_gamma = ks_2samp(data, gamma_rvs)
    bic_gamma = compute_BIC(data, len(gamma_params), gamma_nll)

    return [fisk_nll, ig_nll, ln_nll, weib_nll, gamma_nll,
            ks_fisk, ks_ig, ks_ln, ks_weib, ks_gamma,
            bic_fisk, bic_ig, bic_ln, bic_weib, bic_gamma,
            fisk_params, ig_params, ln_params, weib_params, gamma_params]

def fit_model(args):
    story_id = args[0]
    snaptime = args[1]
    nv = args[2]
    viewTime_array = args[3]

    model_results = run_Parametric(viewTime_array)
    return [story_id, snaptime, nv, model_results]

def run_Models_parallel(story_ids, snaptimes, tv_arr, nv_arr, lthreshold):
    pbar = ProgressBar()
    args = []
    for i in pbar(range(len(story_ids))):
        if nv_arr[i] > lthreshold:
            args.append([story_ids[i], snaptimes[i], nv_arr[i], tv_arr[i]])

    results = Parallel(n_jobs=-1)(map(delayed(fit_model), args))
    return results

if __name__ == '__main__':
    OUT_DIR = "OUT_DIR"
    try:
        os.mkdir(os.path.join(OUT_DIR,"Snaps_LoopTrue"))
    except:
        pass
    
    OUT_DIR = os.path.join(OUT_DIR,"Snaps_LoopTrue")

    print "Loading Overall Data"
    fname = "INPUT FILE in GS bucket containing the snap id, duration and views"
    df = dd.read_csv(fname, dtype={'f0_': 'float64'})
    tv_df = df.compute()

    story_ids = []
    snaptimes = []
    tv_arr = []
    nv_arr = []
    pbar = ProgressBar()

    tview_arr = tv_df['tview_arr'].values
    #snaptime = tv_df['a_snaptime'].values
    snaptime = tv_df['f0_'].values;
    storyIds = tv_df['story_snap_id'].values

    for i in pbar(range(len(tv_df))):
        temp = np.array([float(x) for x in tview_arr[i].split(",")])
        stime = float(snaptime[i])
        if (stime >= 4.0) and (stime < 11.0):
            temp = temp[temp > 0]
            nv_arr.append(temp.shape[0])
            snaptimes.append(float(snaptime[i]))
            tv_arr.append(np.array(temp) / float(snaptime[i]))
            story_ids.append(storyIds[i])

    print len(story_ids), len(snaptimes), len(tv_arr), len(nv_arr)

    story_ids = np.array(story_ids)
    snaptimes = np.array(snaptimes)
    tv_arr = np.array(tv_arr)
    nv_arr = np.array(nv_arr)

    lthreshold = 100
    results = run_Models_parallel(story_ids, snaptimes, tv_arr, nv_arr, lthreshold)
    # out file - pickled file where to dump the fits from all candidate models
    pickle.dump(results,open(out_file, 'w'))
