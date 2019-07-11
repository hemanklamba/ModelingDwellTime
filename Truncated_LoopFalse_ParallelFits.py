import numpy as np
from scipy.stats import gamma, poisson, nbinom, zipf, invgauss, fisk, lognorm, expon, weibull_min
import statsmodels as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import comb, factorial
from scipy import stats
from collections import Counter
import dask.dataframe as dd
from datetime import datetime
import cPickle as pickle
import os
from joblib import Parallel, delayed
from scipy.stats import ks_2samp

class TruncatedFisk_Prior(GenericLikelihoodModel):

    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedFisk_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        prior = params[0]
        c = params[1]
        scale = params[2]

        return -np.log(truncfiskprior_pdf(self.endog, prior, c, scale))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            prior = 0.2
            c = 0.5
            scale = 1.0

            start_params = np.array([prior, c, scale])

        return super(TruncatedFisk_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                             maxfun=maxfun, method='nm', **kwds)

def truncfiskprior_pdf(data, prior, c, scale):
    epsilon = 1e-200
    term1 =  prior * (data == 1.0)
    term2 = (1 - prior) * (fisk.pdf(data, c, loc=0.0, scale=scale)/(
            fisk.cdf(1.0, c, loc=0.0, scale=scale) - fisk.cdf(0.0, c, loc=0.0, scale=scale))) * (data < 1.0)

    return term1 + term2 + epsilon

def truncfiskprior_rvs(prob, c, scale, size):
    prob = max(prob, 1e-10)
    falseEntries = np.zeros((0,))
    failure_ctr = 5;
    while falseEntries.shape[0] < size and failure_ctr > 0:
        s = fisk.rvs(c, loc=0.0, scale=scale, size=size)
        accepted = s[(s <= 1.0)]
        if len(accepted) == 0:
            failure_ctr -=1;
        falseEntries = np.concatenate((falseEntries, accepted), axis=0)
        falseEntries = falseEntries[:size]
    if failure_ctr <= 0:    falseEntries = np.zeros(size);
    indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
    falseEntries[indexes] = 1.0

    return falseEntries


class TruncatedLogNormal_Prior(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedLogNormal_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        prior = params[0]
        mu = params[1]
        sigma = params[2]

        return -np.log(trunclognormprior_pdf(self.endog, prior, mu, sigma))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            prior = 0.2
            mu = 0.5
            sigma = 1.0

            start_params = np.array([prior, mu, sigma])

        return super(TruncatedLogNormal_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                             maxfun=maxfun, method='nm', **kwds)

def trunclognormprior_pdf(data, prior, mu, sigma):
    epsilon = 1e-200
    term1 = prior * (data == 1.0)
    term2 = (1 - prior) * (
                lognorm.pdf(data, sigma, scale=mu, loc=0.0) / (lognorm.cdf(1.0, sigma, scale=mu, loc=0.0) - lognorm.cdf(0.0, sigma, scale=mu, loc=0.0))) * (
                        data < 1.0)

    return term1 + term2 + epsilon

def trunclognormprior_rvs(prob, mu, sigma, size):
    prob = max(1e-10, prob)
    falseEntries = np.zeros((0,))
    failure_ctr = 5;
    while falseEntries.shape[0] < size and failure_ctr > 0:
        s = lognorm.rvs(sigma, scale=mu,loc=0.0, size=size)
        accepted = s[(s <= 1.0)]
        if len(accepted) == 0:
            failure_ctr -=1;
        falseEntries = np.concatenate((falseEntries, accepted), axis=0)
        falseEntries = falseEntries[:size]
    if failure_ctr <= 0:    falseEntries = np.zeros(size);
    indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
    falseEntries[indexes] = 1.0

    return falseEntries


class TruncatedInvGaussian_Prior(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedInvGaussian_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        prior = params[0]
        mu = params[1]
        sigma = params[2]

        return -np.log(truncinvgaussprior_pdf(self.endog, prior, mu, sigma))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            prior = 0.2
            mu = 0.5
            sigma = 1.0

            start_params = np.array([prior, mu, sigma])

        return super(TruncatedInvGaussian_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                             maxfun=maxfun, method='nm', **kwds)

def truncinvgaussprior_pdf(data, prior, mu, sigma):
    epsilon = 1e-200
    term1 = prior * (data == 1.0)
    term2 = (1 - prior) * (
            invgauss.pdf(data, sigma, scale=mu, loc=0.0) / (
                invgauss.cdf(1.0, sigma, scale=mu, loc=0.0) - invgauss.cdf(0.0, sigma, scale=mu, loc=0.0))) * (
                    data < 1.0)

    return term1 + term2 + epsilon

def truncinvgauss_rvs(prob, mu, sigma, size):
    prob = max(1e-10, prob)
    falseEntries = np.zeros((0,))
    failure_ctr = 5;
    while falseEntries.shape[0] < size and failure_ctr > 0:
        s = invgauss.rvs(sigma, scale=mu, loc=0.0, size=size)
        accepted = s[(s <= 1.0)]
        if len(accepted) == 0:
            failure_ctr -= 1;
        falseEntries = np.concatenate((falseEntries, accepted), axis=0)
        falseEntries = falseEntries[:size]
    if failure_ctr <= 0:    falseEntries = np.zeros(size);
    indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
    falseEntries[indexes] = 1.0

    return falseEntries


class TruncatedWeibull_Prior(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedWeibull_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        prior = params[0]
        c = params[1]
        scale = params[2]

        return -np.log(truncatedweibull_pdf(self.endog, prior, c, scale))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            prior = 0.1
            c = 0.5
            scale = 1.0

            start_params = np.array([prior, c, scale])

        return super(TruncatedWeibull_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                                       maxfun=maxfun, method='nm', **kwds)

def truncatedweibull_pdf(data, prior, c, scale):
    epsilon = 1e-200
    term1 = prior * (data == 1.0)
    term2 = (1 - prior) * (
            weibull_min.pdf(data, c, scale=scale, loc=0.0) / (
                weibull_min.cdf(1.0, c, scale=scale, loc=0.0) - weibull_min.cdf(0.0, c, scale=scale, loc=0.0))) * (
                    data < 1.0)

    return term1 + term2 + epsilon

def truncweibull_rvs(prob, c, scale, size):
    prob = max(1e-10, prob)
    falseEntries = np.zeros((0,))
    failure_ctr = 5;
    while falseEntries.shape[0] < size and failure_ctr > 0:
        s = weibull_min.rvs(c, scale=scale, loc=0.0, size=size)
        accepted = s[(s <= 1.0)]
        if len(accepted) <= 0:
            failure_ctr -=1;
        falseEntries = np.concatenate((falseEntries, accepted), axis=0)
        falseEntries = falseEntries[:size]
    if failure_ctr <= 0:    falseEntries = np.zeros(size);
    indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
    falseEntries[indexes] = 1.0

    return falseEntries


class TruncatedExpon_Prior(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedExpon_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        prior = params[0]
        c = params[1]

        return -np.log(truncexponprior_pdf(self.endog, prior, c))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            prior = 0.2
            c = 0.5

            start_params = np.array([prior, c])

        return super(TruncatedExpon_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                             maxfun=maxfun, method='nm', **kwds)

def truncexponprior_pdf(data, prior, c):
    epsilon = 1e-200
    term1 = prior * (data == 1.0)
    term2 = (1 - prior) * (
            expon.pdf(data, scale=c, loc=0.0) / (
                expon.cdf(1.0, scale=c, loc=0.0) - expon.cdf(0.0, scale=c, loc=0.0))) * (data < 1.0)

    return term1 + term2 + epsilon

def truncexponprior_rvs(prob, c, size):
    prob = max(0, prob)
    entries = np.empty([size,])
    for i in range(size):
        temp = expon.rvs(scale=c, size=1)

        while temp >= 1.0:
            temp = expon.rvs(scale=c, size=1)

        entries[i] = temp

    indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
    print indexes.shape
    entries[indexes] = 1.0

    return entries


class TruncatedGamma_Prior(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedGamma_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        prior = params[0]
        c = params[1]
        scale = params[2]

        return -np.log(truncgammaprior_pdf(self.endog, prior, c, scale))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            prior = 0.2
            c = 0.5
            scale = 1.0

            start_params = np.array([prior, c, scale])

        return super(TruncatedGamma_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                             maxfun=maxfun, method='nm', **kwds)

def truncgammaprior_pdf(data, prior, c, scale):
    epsilon = 1e-200
    term1 = prior * (data == 1.0)
    term2 = (1 - prior) * (
            gamma.pdf(data, c, scale=scale, loc=0.0) / (
                gamma.cdf(1.0, c, scale=scale, loc=0.0) - gamma.cdf(0.0, c, scale=scale, loc=0.0))) * (
                    data < 1.0)

    return term1 + term2 + epsilon

def truncgamma_rvs(prob, c, scale, size):
    prob = max(1e-10, prob)
    falseEntries = np.zeros((0,))
    failure_ctr = 5;
    while falseEntries.shape[0] < size and failure_ctr > 0:
        s = gamma.rvs(c, scale=scale, loc=0.0, size=size)
        accepted = s[(s <= 1.0)]
        if len(accepted) <= 0:
            failure_ctr -=1;
        falseEntries = np.concatenate((falseEntries, accepted), axis=0)
        falseEntries = falseEntries[:size]
    if failure_ctr <= 0:    falseEntries = np.zeros(size);
    indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
    falseEntries[indexes] = 1.0

    return falseEntries


def compute_LogLikelihood(data, fisk_params, ig_params, lognorm_params, gamma_params):
    n_fisk_params = 2
    n_ig_params = 3
    n_lognorm_params = 3
    n_gamma_params = 3

    trunc_data = data[data!=1.0]
    prob = np.sum(data == 1.0)/float(data.shape[0])

    ll_fisk_trunc = np.sum(-np.log(fisk.pdf(trunc_data, *fisk_params)))
    bic = np.log(trunc_data.shape[0]) * n_fisk_params - 2 * ll_fisk_trunc
    #ll_fisk_nontrunc =  np.sum(-np.log(prob * np.ones()))

    ll_ig_trunc = np.sum(-np.log(invgauss.pdf(trunc_data, *ig_params)))
    #ll_ig_nontrunc = np.sum(np.log(prob))

    ll_ln_trunc = np.sum(-np.log(lognorm.pdf(trunc_data, *lognorm_params)))
    #ll_ln_nontrunc = np.sum(np.log(prob))

    ll_gamma_trunc = np.sum(-np.log(gamma.pdf(trunc_data, *gamma_params)))
    #ll_exp_trunc = np.sum(-np.log(expon.pdf(trunc_data, *exp_params)))
    #ll_exp_nontrunc = np.sum(np.log(prob))

    return ll_fisk_trunc, ll_ig_trunc, ll_ln_trunc, ll_gamma_trunc

def compute_LogLikelihood_ExtData(data, fisk_params, ig_params):
    prob = fisk_params[0]
    fisk_params = fisk_params[1]
    num_one =  np.sum(data == 1.0)

    trunc_data = data[data!=1.0]

    fisk_trunc_ll = np.sum(-np.log(fisk.pdf(trunc_data, *fisk_params)))
    fisk_nontrunc_ll = -np.log(prob)*num_one

    fisk_ll = fisk_trunc_ll + fisk_nontrunc_ll

    ig_trunc_ll = np.sum(-np.log(invgauss.pdf(trunc_data, *fisk_params)))
    ig_nontrunc_ll = -np.log(prob) * num_one

    ig_ll = ig_trunc_ll + ig_nontrunc_ll

    return fisk_ll, ig_ll

def check_expon():
    data = expon.rvs(0.2, size=1000)
    data[data>1.0] = 1.0
    print "Orig Frac=" + str(np.sum(data == 1.0) / float(data.shape[0]))
    rv = TruncatedExpon_Prior(data)
    res = rv.fit()
    print res.params
    print "================================"

    x =  truncexponprior_rvs(res.params[0], res.params[1], size=1000)
    print min(x), max(x)
    print "Frac="+str(np.sum(x==1.0)/float(np.sum(x)))

def check_invgaussian():
    data = invgauss.rvs(0.5, scale=1.0, size=1000)
    data[data>1.0] = 1.0
    print "Orig Frac="+str(np.sum(data==1.0)/float(data.shape[0]))
    rv = TruncatedInvGaussian_Prior(data)
    res = rv.fit()
    print res.params
    print "================================"
    x = truncinvgauss_rvs(res.params[0], res.params[1], res.params[2], size=1000)
    print min(x), max(x)
    print "Frac=" + str(np.sum(x == 1.0) / float((x.shape[0])))

def check_fiskprior():
    data = fisk.rvs(c=0.5, scale=1.0, size=1000)
    data[data>1.0] = 1.0
    print "Orig Frac="+str(np.sum(data==1.0)/float(data.shape[0]))
    rv = TruncatedFisk_Prior(data)
    res = rv.fit()
    print res.params
    print "================================"
    x = truncfiskprior_rvs(res.params[0], res.params[1], res.params[2], size=1000)
    print min(x), max(x)
    print "Frac=" + str(np.sum(x == 1.0) / float((x.shape[0])))

def check_lognorm():
    data = lognorm.rvs(0.5, scale=1.0, loc=0.0, size=10000)
    data[data>1.0] = 1.0
    print "Orig Frac="+str(np.sum(data==1.0)/float(data.shape[0]))
    rv = TruncatedLogNormal_Prior(data)
    res = rv.fit()
    print res.params
    print "================================"
    x = trunclognormprior_rvs(res.params[0], res.params[1], res.params[2], size=1000)
    print min(x), max(x)
    print "Frac=" + str(np.sum(x == 1.0) / float((x.shape[0])))

def compute_BIC(data, n_params, neg_ll):
    bic = (np.log(data.shape[0]) * n_params) - 2 * -neg_ll
    return bic

def fit_truncated_distributions(arg_arr):
    story_id = arg_arr[0]
    snapTime = arg_arr[1]
    data = arg_arr[2]

    distribution = TruncatedFisk_Prior
    rv_fisk = distribution(data)
    res_fisk = rv_fisk.fit()
    neg_ll_fisk = np.sum(rv_fisk.nloglikeobs(res_fisk.params))
    bic_fisk = compute_BIC(data, len(res_fisk.params), neg_ll_fisk)
    rvs_fisk = truncfiskprior_rvs(res_fisk.params[0], res_fisk.params[1], res_fisk.params[2], data.shape[0])
    ksResults_fisk = ks_2samp(data, rvs_fisk)

    distribution = TruncatedInvGaussian_Prior
    rv_ig = distribution(data)
    res_ig = rv_ig.fit()
    neg_ll_ig = np.sum(rv_ig.nloglikeobs(res_ig.params))
    bic_ig = compute_BIC(data, len(res_ig.params), neg_ll_ig)
    rvs_ig = truncinvgauss_rvs(res_ig.params[0], res_ig.params[1], res_ig.params[2], data.shape[0])
    ksResults_ig = ks_2samp(data, rvs_ig)

    distribution =  TruncatedLogNormal_Prior
    rv_ln = distribution(data)
    res_ln = rv_ln.fit()
    neg_ll_ln = np.sum(rv_ln.nloglikeobs(res_ln.params))
    bic_ln = compute_BIC(data, len(res_ln.params), neg_ll_ln)
    rvs_ln = trunclognormprior_rvs(res_ln.params[0], res_ln.params[1], res_ln.params[2], data.shape[0])
    ksResults_ln = ks_2samp(data, rvs_ln)

    distribution = TruncatedWeibull_Prior
    rv_weib = distribution(data)
    res_weib = rv_weib.fit()
    neg_ll_weib = np.sum(rv_weib.nloglikeobs(res_weib.params))
    bic_weib = compute_BIC(data, len(res_weib.params), neg_ll_weib)
    rvs_weib = truncweibull_rvs(res_weib.params[0], res_weib.params[1], res_weib.params[2], data.shape[0])
    ksResults_weib = ks_2samp(data, rvs_weib)

    distribution = TruncatedGamma_Prior
    rv_gamma = distribution(data)
    res_gamma = rv_gamma.fit()
    neg_ll_gamma = np.sum(rv_gamma.nloglikeobs(res_gamma.params))
    bic_gamma = compute_BIC(data, len(res_gamma.params), neg_ll_gamma)
    rvs_gamma = truncgamma_rvs(res_gamma.params[0], res_gamma.params[1], res_gamma.params[2], data.shape[0])
    ksResults_gamma = ks_2samp(data, rvs_gamma)

    return [story_id, snapTime, data.shape[0], neg_ll_fisk, neg_ll_ig, neg_ll_ln, neg_ll_weib, neg_ll_gamma,
            ksResults_fisk, ksResults_ig, ksResults_ln, ksResults_weib, ksResults_gamma,
            bic_fisk, bic_ig, bic_ln, bic_weib, bic_gamma,
            res_fisk.params, res_ig.params, res_ln.params, res_weib.params, res_gamma.params]

def parallelParamFit(lthreshold):
    fname = "INPUT_FILE_stored_in_GS_storage_file - containing - id of story, duration of story, and the view times for every view"
    df = dd.read_csv(fname)
    tv_df = df.compute()
    print tv_df.shape
    print tv_df.columns

    story_ids = tv_df['story_snap_id'].values
    snap_times = tv_df['f0_'].values
    viewed_times = tv_df['tview_arr'].values

    non_nan = ~np.isnan(snap_times)
    story_ids = story_ids[non_nan]
    snap_times = snap_times[non_nan]
    viewed_times = viewed_times[non_nan]

    lt10 = snap_times < 11

    story_ids = story_ids[lt10]
    snap_times = snap_times[lt10]
    viewed_times = viewed_times[lt10]

    args = []
    for i in range(story_ids.shape[0]):
        story_id = story_ids[i]
        snapTime = snap_times[i]

        viewTime_arr = np.array([float(x) for x in viewed_times[i].split(",")])
        viewTime_arr[viewTime_arr > snapTime] = snapTime
        viewTime_arr = viewTime_arr[viewTime_arr > 0]

        binned_snapTime = int(snapTime)

        if binned_snapTime >= 4:
            viewTime_arr = viewTime_arr / snapTime
            if viewTime_arr.shape[0] > lthreshold:
                args.append([story_id, snapTime, viewTime_arr])

    #results = Parallel(n_jobs=-1)(map(delayed(fit_0loc_distributions), args))
    results = Parallel(n_jobs=-1)(map(delayed(fit_truncated_distributions), args))
    return results

if __name__ == '__main__':
    # Enter output directory here
    OUT_DIR = "OUT_DIR"
    try:
        os.mkdir(os.path.join(OUT_DIR, "Snaps_LoopFalse"))
    except:
        pass;

    OUT_DIR = os.path.join(OUT_DIR, "Snaps_LoopFalse")

    lthreshold = 100
    results = parallelParamFit(lthreshold)
    # out_file - pickled file containing the fits from all candidate models
    pickle.dump(results, open(out_file,'w'))

