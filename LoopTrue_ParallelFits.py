import os
import sys
import numpy as np
import pandas as pd
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

def run_Parametric(story_id, data):
    print "["+str(story_id)+"]Fitting Fisk"
    fisk_params = fisk.fit(data, floc=0)
    fisk_nll = fisk.nnlf(fisk_params, data)
    fisk_rvs = fisk.rvs(*fisk_params, size=data.shape[0])
    ks_fisk = ks_2samp(data, fisk_rvs)
    bic_fisk = compute_BIC(data, len(fisk_params), fisk_nll)

    print "[" + str(story_id) + "]Fitting IG"
    ig_params = invgauss.fit(data, floc=0)
    ig_nll = invgauss.nnlf(ig_params, data)
    ig_rvs = invgauss.rvs(*ig_params, size=data.shape[0])
    ks_ig = ks_2samp(data, ig_rvs)
    bic_ig = compute_BIC(data, len(ig_params), ig_nll)

    print "[" + str(story_id) + "]Fitting LN"
    ln_params = lognorm.fit(data, floc=0)
    ln_nll = lognorm.nnlf(ln_params, data)
    ln_rvs = lognorm.rvs(*ln_params, size=data.shape[0])
    ks_ln = ks_2samp(data, ln_rvs)
    bic_ln = compute_BIC(data, len(ln_params), ln_nll)

    print "[" + str(story_id) + "]Fitting Weibull"
    weib_params = weibull_min.fit(data, floc=0)
    weib_nll = weibull_min.nnlf(weib_params, data)
    weib_rvs = weibull_min.rvs(*weib_params, size=data.shape[0])
    ks_weib = ks_2samp(data, weib_rvs)
    bic_weib = compute_BIC(data, len(weib_params), weib_nll)

    print "[" + str(story_id) + "]Fitting Gamma"
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
    nv = args[1]
    viewTime_array = args[2]

    model_results = run_Parametric(story_id,viewTime_array)
    return [story_id, nv, model_results]

def run_Models_parallel(story_ids, tv_arr, nv_arr, lthreshold):
    print(story_ids[0])
    print(tv_arr[0])
    print(nv_arr[0])
    pbar = ProgressBar()
    args = []
    for i in pbar(range(len(story_ids))):
        if nv_arr[i] > lthreshold:
            args.append([story_ids[i], nv_arr[i], tv_arr[i]])

    results = Parallel(n_jobs=-1)(map(delayed(fit_model), args))
    return results

def read_file(in_file):
    story_ids = []
    tv_arr = []
    nv_arr = []

    f = open(in_file, 'r')
    line = f.readline()
    while line:
        line = line.replace("\n", "")
        split = line.split("\t")
        story_id = split[0]
        story_ids.append(story_id)
        split = split[1].split(",")
        arr_views = []
        for i in range(len(split)):
            arr_views.append(float(split[i]))
        arr_views = np.array(arr_views)
        tv_arr.append(arr_views)
        nv_arr.append(len(arr_views))
        line = f.readline()
    f.close()

    return story_ids, tv_arr, nv_arr

if __name__ == '__main__':
    OUT_DIR = "RESULTS_SAMPLE"
    try:
        os.mkdir(os.path.join(OUT_DIR,"Snaps_LoopTrue"))
    except:
        pass
    
    OUT_DIR = os.path.join(OUT_DIR,"Snaps_LoopTrue")


    print "Loading Overall Data"
    fname = "DATA/Sample_LoopTrue_Views.txt"
    story_ids, tv_arr, nv_arr = read_file(fname)


    pbar = ProgressBar()

    story_ids = np.array(story_ids)
    tv_arr = np.array(tv_arr)
    nv_arr = np.array(nv_arr)

    lthreshold = 100
    results = run_Models_parallel(story_ids, tv_arr, nv_arr, lthreshold)
    out_file = os.path.join(OUT_DIR, "LoopTrue.pkl")
    pickle.dump(results,open(out_file, 'w'))
