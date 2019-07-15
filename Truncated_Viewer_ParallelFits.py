import os, sys
import numpy as np
from scipy.stats import fisk, lognorm, powerlaw, invgauss, genpareto, expon, pareto, weibull_min, gamma
from scipy.stats import ks_2samp
from statsmodels.base.model import GenericLikelihoodModel
from joblib import Parallel, delayed
import cPickle as pickle
import dask.dataframe as dd
import time

'''
LOGNORMAL
'''

class TruncatedLogNormal_Prior(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedLogNormal_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        mu = params[0]
        sigma = params[1]

        return -np.log(trunclognormprior_pdf(self.endog, mu, sigma))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            mu = 0.5
            sigma = 1.0

            start_params = np.array([mu, sigma])

        return super(TruncatedLogNormal_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                             maxfun=maxfun, method='nm', **kwds)

def trunclognormprior_pdf(data, mu, sigma):
    epsilon = 1e-200
    term2 = (
                lognorm.pdf(data, sigma, scale=mu, loc=0.0) / (lognorm.cdf(1.0, sigma, scale=mu, loc=0.0) - lognorm.cdf(0.0, sigma, scale=mu, loc=0.0))) * (
                        data < 1.0)

    return  term2 + epsilon

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
    if size > 0:
        indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
        falseEntries[indexes] = 1.0
    return falseEntries

def fit_lognormal_loop(data):
    loop_false = (data > 0)
    loop_true = (data < 0)

    data_false = data[loop_false]
    data_true = data[loop_true]

    loop_prob = np.sum(loop_false) / float(data.shape[0])

    ln_trunc_fit = np.array([np.nan, 0.0, np.nan])

    nll_false = 0
    nll_true = 0

    # Fitting FalseLoop Data
    if np.sum(loop_false)>0:
        trunc_data = data_false[data_false<1.0]
        prior = (len(data_false) - len(trunc_data)) / float(len(data_false))
        if trunc_data.shape[0] > 0:
            distribution = TruncatedLogNormal_Prior
            rv_lognormal_false = distribution(trunc_data)
            res_lognormal_false = rv_lognormal_false.fit()
            ln_trunc_fit = np.array([prior, res_lognormal_false.params[0], res_lognormal_false.params[1]])

            nll_false = np.sum(rv_lognormal_false.nloglikeobs(res_lognormal_false.params))

    # Fitting TrueLoop Data
    if np.sum(loop_true) > 0:
        ln_trueloop_fit = lognorm.fit(np.abs(data_true), floc=0.0)
        nll_true = -np.sum(np.log(lognorm.pdf(np.abs(data_true), ln_trueloop_fit[0], ln_trueloop_fit[1], ln_trueloop_fit[2])+1e-200))
    else:
        ln_trueloop_fit = np.array([np.nan, 0.0, np.nan])

    nll = nll_false+nll_true
    return [loop_prob, ln_trunc_fit, ln_trueloop_fit, nll]

def lognorm_rvs(ln_params, size):
    # Parameters
    # [loop_prob, ln_trunc_fit, ln_trueloop_fit, nll]

    num_falseLoop = int(round(ln_params[0] * size))
    num_trueLoop = int((1 - ln_params[0]) * size)

    falseEntries = np.array([])
    trueEntries = np.array([])

    if ~np.isnan(ln_params[1][1]) and (ln_params[1][1]!=0.0):
        falseEntries = np.zeros((0,))
        falseEntries = trunclognormprior_rvs(ln_params[1][0], ln_params[1][1], ln_params[1][2], size = num_falseLoop)

    if ~np.isnan(ln_params[2][0]) and (ln_params[2][0]!=0.0):
        trueEntries = lognorm.rvs(ln_params[2][0], ln_params[2][1], ln_params[2][2], size=num_trueLoop)

    if (~np.isnan(ln_params[1][1])) and (~np.isnan(ln_params[2][0])):
        entries = np.concatenate((falseEntries, -trueEntries), axis=0)
    elif ~np.isnan(ln_params[1][1]):
        entries = falseEntries
    elif ~np.isnan(ln_params[2][0]):
        entries = -trueEntries
    else:
        entries = np.array([1.0] * size);

    print entries.shape[0]
    return entries

'''
INV GAUSSIAN
'''

class TruncatedInvGaussian_Prior(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedInvGaussian_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        mu = params[0]
        sigma = params[1]

        return -np.log(truncinvgaussprior_pdf(self.endog, mu, sigma))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            mu = 0.5
            sigma = 1.0

            start_params = np.array([mu, sigma])

        return super(TruncatedInvGaussian_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                             maxfun=maxfun, method='nm', **kwds)

def truncinvgaussprior_pdf(data, mu, sigma):
    epsilon = 1e-200
    term2 =  (
            invgauss.pdf(data, sigma, scale=mu, loc=0.0) / (
                invgauss.cdf(1.0, sigma, scale=mu, loc=0.0) - invgauss.cdf(0.0, sigma, scale=mu, loc=0.0))) * (
                    data < 1.0)

    return term2 + epsilon

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
    if size > 0:
        indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
        falseEntries[indexes] = 1.0
    return falseEntries

def fit_ig_loop(data):
    loop_false = (data > 0)
    loop_true = (data < 0)

    data_false = data[loop_false]
    data_true = data[loop_true]

    loop_prob = np.sum(loop_false) / float(data.shape[0])
    ig_trunc_fit = np.array([np.nan, 0.0, np.nan])
    
    nll_false = 0;
    nll_true = 0;

    if np.sum(loop_false) > 0:
        trunc_data = data_false[data_false < 1.0]
        prior = (len(data_false) - len(trunc_data)) / float(len(data_false))
        if trunc_data.shape[0] > 0:
            distribution = TruncatedInvGaussian_Prior
            rv_ig_false = distribution(trunc_data)
            res_ig_false = rv_ig_false.fit()
            ig_trunc_fit = np.array([prior, res_ig_false.params[0], res_ig_false.params[1]])
            nll_false = np.sum(rv_ig_false.nloglikeobs(res_ig_false.params))

    if np.sum(loop_true) > 0:
        ig_trueloop_fit = invgauss.fit(np.abs(data_true), floc=0.0)
        nll_true = -np.sum(
            np.log(lognorm.pdf(np.abs(data_true), ig_trueloop_fit[0], ig_trueloop_fit[1], ig_trueloop_fit[2]) + 1e-200))
    else:
        ig_trueloop_fit = np.array([np.nan, 0.0, np.nan])

    nll = nll_false+nll_true

    return [loop_prob, ig_trunc_fit, ig_trueloop_fit, nll]

def ig_rvs(ig_params, size):
    num_falseLoop = int(ig_params[0] * size)
    num_trueLoop = int((1 - ig_params[0]) * size)

    falseEntries = np.array([])
    trueEntries = np.array([])

    if ~np.isnan(ig_params[1][1]) and (ig_params[1][1]!=0):
        falseEntries = np.zeros((0,))
        falseEntries = truncinvgauss_rvs(ig_params[1][0], ig_params[1][1], ig_params[1][2], size=num_falseLoop)

    if ~np.isnan(ig_params[2][0]) and (ig_params[2][0]!=0):
        trueEntries = invgauss.rvs(ig_params[2][0], ig_params[2][1], ig_params[2][2], size=num_trueLoop)
        print trueEntries.shape

    if (~np.isnan(ig_params[1][1])) and (~np.isnan(ig_params[2][0])):
        entries = np.concatenate((falseEntries, -trueEntries), axis=0)
    elif ~np.isnan(ig_params[1][1]):
        entries = falseEntries
    elif ~np.isnan(ig_params[2][0]):
        entries = -trueEntries
    else:
        entries = np.array([1.0] * size);

    print entries.shape[0]
    return entries

'''
FISK
'''

class TruncatedFisk_Prior(GenericLikelihoodModel):

    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedFisk_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        c = params[0]
        scale = params[1]

        return -np.log(truncfiskprior_pdf(self.endog, c, scale))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            c = 0.5
            scale = 1.0

            start_params = np.array([c, scale])

        return super(TruncatedFisk_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                             maxfun=maxfun, method='nm', **kwds)

def truncfiskprior_pdf(data, c, scale):
    epsilon = 1e-200
    term2 = (fisk.pdf(data, c, loc=0.0, scale=scale)/(
            fisk.cdf(1.0, c, loc=0.0, scale=scale) - fisk.cdf(0.0, c, loc=0.0, scale=scale))) * (data < 1.0)

    return term2 + epsilon

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
    if size > 0:
        indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
        falseEntries[indexes] = 1.0
    return falseEntries

def fit_fisk_loop(data):
    loop_false = (data > 0)
    loop_true = (data < 0)

    data_false = data[loop_false]
    data_true = data[loop_true]

    loop_prob = np.sum(loop_false) / float(data.shape[0])
    fisk_trunc_fit = np.array([np.nan, 0.0, np.nan])

    nll_false = 0
    nll_true = 0

    if np.sum(loop_false) > 0:
        trunc_data = data_false[data_false < 1.0]
        prior = (len(data_false) - len(trunc_data)) / float(len(data_false))
        if trunc_data.shape[0] > 0:
            distribution = TruncatedFisk_Prior
            rv_fisk_false = distribution(trunc_data)
            res_fisk_false = rv_fisk_false.fit()
            fisk_trunc_fit = np.array([prior, res_fisk_false.params[0], res_fisk_false.params[1]])

            nll_false = np.sum(rv_fisk_false.nloglikeobs(res_fisk_false.params))

    if np.sum(loop_true) > 0:
        fisk_trueloop_fit = fisk.fit(np.abs(data_true), floc=0.0)
        nll_true = -np.sum(
            np.log(lognorm.pdf(np.abs(data_true), fisk_trueloop_fit[0], fisk_trueloop_fit[1], fisk_trueloop_fit[2]) + 1e-200))
    else:
        fisk_trueloop_fit = np.array([np.nan, 0.0, np.nan])

    nll = nll_false+nll_true

    return [loop_prob, fisk_trunc_fit, fisk_trueloop_fit, nll]

def fisk_rvs(fisk_params, size):
    num_falseLoop = int(fisk_params[0] * size)
    num_trueLoop = int((1 - fisk_params[0]) * size)

    falseEntries = np.array([])
    trueEntries = np.array([])

    if ~np.isnan(fisk_params[1][1]) and (fisk_params[1][1]!=0.0):
        falseEntries = np.zeros((0,))
        falseEntries = truncfiskprior_rvs(fisk_params[1][0], fisk_params[1][1], fisk_params[1][2], size=num_falseLoop)

    if ~np.isnan(fisk_params[2][0]) and (fisk_params[2][0]!=0):
        trueEntries = invgauss.rvs(fisk_params[2][0], fisk_params[2][1], fisk_params[2][2], size=num_trueLoop)
        print trueEntries.shape

    if (~np.isnan(fisk_params[1][1])) and (~np.isnan(fisk_params[2][0])):
        entries = np.concatenate((falseEntries, -trueEntries), axis=0)
    elif ~np.isnan(fisk_params[1][1]):
        entries = falseEntries
    elif ~np.isnan(fisk_params[2][0]):
        entries = -trueEntries
    else:
        entries = np.array([1.0] * size);

    print entries.shape[0]
    return entries


'''
WEIBULL
'''

class TruncatedWeibull_Prior(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedWeibull_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        c = params[0]
        scale = params[1]

        return -np.log(truncatedweibull_pdf(self.endog, c, scale))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            c = 0.5
            scale = 1.0

            start_params = np.array([c, scale])

        return super(TruncatedWeibull_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                                       maxfun=maxfun, method='nm', **kwds)

def truncatedweibull_pdf(data, c, scale):
    epsilon = 1e-200
    term2 =  (
            weibull_min.pdf(data, c, scale=scale, loc=0.0) / (
                weibull_min.cdf(1.0, c, scale=scale, loc=0.0) - weibull_min.cdf(0.0, c, scale=scale, loc=0.0))) * (
                    data < 1.0)

    return term2 + epsilon

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
    if size > 0:
        indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
        falseEntries[indexes] = 1.0
    return falseEntries

def fit_weibull_loop(data):
    loop_false = (data > 0)
    loop_true = (data < 0)

    data_false = data[loop_false]
    data_true = data[loop_true]

    loop_prob = np.sum(loop_false) / float(data.shape[0])
    weib_trunc_fit = np.array([np.nan, 0.0, np.nan])

    nll_false = 0
    nll_true = 0

    if np.sum(loop_false) > 0:
        trunc_data = data_false[data_false < 1.0]
        prior = (len(data_false) - len(trunc_data)) / float(len(data_false))
        if trunc_data.shape[0] > 0:
            distribution = TruncatedWeibull_Prior
            rv_weib_false = distribution(trunc_data)
            res_weib_false = rv_weib_false.fit()
            weib_trunc_fit = np.array([prior, res_weib_false.params[0], res_weib_false.params[1]])

            nll_false = np.sum(rv_weib_false.nloglikeobs(res_weib_false.params))


    if np.sum(loop_true) > 0:
        weib_trueloop_fit = weibull_min.fit(np.abs(data_true), floc=0.0)
        nll_true = -np.sum(
            np.log(lognorm.pdf(np.abs(data_true), weib_trueloop_fit[0], weib_trueloop_fit[1], weib_trueloop_fit[2]) + 1e-200))
    else:
        weib_trueloop_fit = np.array([np.nan, 0.0, np.nan])

    nll = nll_false+nll_true

    return [loop_prob, weib_trunc_fit, weib_trueloop_fit, nll]

def weibull_ll_loop(params, data):
    loop_prob = params[0]
    complete_prob = params[1]
    ln_false = params[2]
    ln_true = params[3]

    false_pdf1 = np.zeros(data.shape)
    false_pdf2 = np.zeros(data.shape)
    true_pdf = np.zeros(data.shape)

    if ~np.isnan(ln_false[0]):
        false_pdf1 = loop_prob * complete_prob * (data == 1.0)
        false_pdf2 = loop_prob * (1 - complete_prob) * weibull_min.pdf(data, ln_false[0], ln_false[1], ln_false[2]) * (
                data > 0.0) * (data < 1.0)

    if ~np.isnan(ln_true[0]):
        true_pdf = (1 - loop_prob) * weibull_min.pdf(np.abs(data), ln_true[0], ln_true[1], ln_true[2]) * (data < 0.0)

    pdf = false_pdf1 + false_pdf2 + true_pdf + 1e-200
    ll = -np.sum(np.log(pdf))

    return ll

def weibull_rvs(weibull_params, size):
    num_falseLoop = int(weibull_params[0] * size)
    num_trueLoop = int((1 - weibull_params[0]) * size)

    falseEntries = np.array([])
    trueEntries = np.array([])

    if ~np.isnan(weibull_params[1][1]) and (weibull_params[1][1]!=0.0):
        falseEntries = np.zeros((0,))
        falseEntries = truncweibull_rvs(weibull_params[1][0], weibull_params[1][1], weibull_params[1][2], size=num_falseLoop)

    if ~np.isnan(weibull_params[2][0]) and (weibull_params[2][0]!=0.0):
        trueEntries = weibull_min.rvs(weibull_params[2][0], weibull_params[2][1], weibull_params[2][2], size=num_trueLoop)
        print trueEntries.shape

    if (~np.isnan(weibull_params[1][1])) and (~np.isnan(weibull_params[2][0])):
        entries = np.concatenate((falseEntries, -trueEntries), axis=0)
    elif ~np.isnan(weibull_params[1][1]):
        entries = falseEntries
    elif ~np.isnan(weibull_params[2][0]):
        entries = -trueEntries
    else:
        entries = np.array([1.0] * size);

    print entries.shape[0]
    return entries


'''
GAMMA
'''
class TruncatedGamma_Prior(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(TruncatedGamma_Prior, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        c = params[0]
        scale = params[1]

        return -np.log(truncgammaprior_pdf(self.endog, c, scale))

    def fit(self, start_params = None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            c = 0.5
            scale = 1.0

            start_params = np.array([c, scale])

        return super(TruncatedGamma_Prior, self).fit(start_params=start_params, maxiter=maxiter,
                                             maxfun=maxfun, method='nm', **kwds)

def truncgammaprior_pdf(data, c, scale):
    epsilon = 1e-200
    term2 = (gamma.pdf(data, c, scale=scale, loc=0.0) / (
                    gamma.cdf(1.0, c, scale=scale, loc=0.0) - gamma.cdf(0.0, c, scale=scale, loc=0.0))) * (
                    data < 1.0)

    return term2 + epsilon

def truncgamma_rvs(prob, c, scale, size):
    prob = max(1e-10, prob)
    falseEntries = np.zeros((0,))
    failure_ctr = 5;
    while falseEntries.shape[0] < size and failure_ctr > 0:
        s = gamma.rvs(c, scale=scale, loc=0.0, size=size)
        accepted = s[(s <= 1.0)]
        if len(accepted) <= 0:
            failure_ctr -= 1;
        falseEntries = np.concatenate((falseEntries, accepted), axis=0)
        falseEntries = falseEntries[:size]
    if failure_ctr <= 0:    falseEntries = np.zeros(size);
    if size > 0:
        indexes = np.random.choice(range(size), size=int(prob * size), replace=False)
        falseEntries[indexes] = 1.0
    return falseEntries

def fit_gamma_loop(data):
    loop_false = (data > 0)
    loop_true = (data < 0)

    data_false = data[loop_false]
    data_true = data[loop_true]

    loop_prob = np.sum(loop_false) / float(data.shape[0])
    gamma_trunc_fit = np.array([np.nan, 0.0, np.nan])

    nll_false = 0
    nll_true = 0

    if np.sum(loop_false) > 0:
        trunc_data = data_false[data_false < 1.0]
        prior = (len(data_false) - len(trunc_data)) / float(len(data_false))
        if trunc_data.shape[0] > 0:
            distribution = TruncatedGamma_Prior
            rv_gamma_false = distribution(trunc_data)
            res_gamma_false = rv_gamma_false.fit()
            gamma_trunc_fit = np.array([prior, res_gamma_false.params[0], res_gamma_false.params[1]])
            nll_false = np.sum(rv_gamma_false.nloglikeobs(res_gamma_false.params))

    if np.sum(loop_true) > 0:
        gamma_trueloop_fit = gamma.fit(np.abs(data_true), floc=0.0)
        nll_true = -np.sum(
            np.log(lognorm.pdf(np.abs(data_true), gamma_trueloop_fit[0], gamma_trueloop_fit[1], gamma_trueloop_fit[2]) + 1e-200))
    else:
        gamma_trueloop_fit = np.array([np.nan, 0.0, np.nan])

    nll = nll_false+nll_true

    return [loop_prob, gamma_trunc_fit, gamma_trueloop_fit, nll]

def gamma_ll_loop(params, data):
    loop_prob = params[0]
    complete_prob = params[1]
    ln_false = params[2]
    ln_true = params[3]

    false_pdf1 = np.zeros(data.shape)
    false_pdf2 = np.zeros(data.shape)
    true_pdf = np.zeros(data.shape)

    if ~np.isnan(ln_false[0]):
        false_pdf1 = loop_prob * complete_prob * (data == 1.0)
        false_pdf2 = loop_prob * (1 - complete_prob) * gamma.pdf(data, ln_false[0], ln_false[1], ln_false[2]) * (
                data > 0.0) * (data < 1.0)

    if ~np.isnan(ln_true[0]):
        true_pdf = (1 - loop_prob) * gamma.pdf(np.abs(data), ln_true[0], ln_true[1], ln_true[2]) * (data < 0.0)

    pdf = false_pdf1 + false_pdf2 + true_pdf + 1e-200
    ll = -np.sum(np.log(pdf))

    return ll

def gamma_rvs(gamma_params, size):
    num_falseLoop = int(gamma_params[0] * size)
    num_trueLoop = int((1 - gamma_params[0]) * size)

    falseEntries = np.array([])
    trueEntries = np.array([])

    if ~np.isnan(gamma_params[1][1]) and (gamma_params[1][1]!=0.0):
        falseEntries = np.zeros((0,))
        falseEntries = truncgamma_rvs(gamma_params[1][0], gamma_params[1][1], gamma_params[1][2], size=num_falseLoop)

    if ~np.isnan(gamma_params[2][0]) and (gamma_params[2][0]!=0.0):
        trueEntries = gamma.rvs(gamma_params[2][0], gamma_params[2][1], gamma_params[2][2], size=num_trueLoop)
        print trueEntries.shape

    if (~np.isnan(gamma_params[1][1])) and (~np.isnan(gamma_params[2][0])):
        entries = np.concatenate((falseEntries, -trueEntries), axis=0)
    elif ~np.isnan(gamma_params[1][1]):
        entries = falseEntries
    elif ~np.isnan(gamma_params[2][0]):
        entries = -trueEntries
    else:
        entries = np.array([1.0] * size);

    print entries.shape[0]
    return entries

def compute_LogLikelihood(data, fisk_params, ig_params, lognorm_params, exp_params, gamma_params):
    trunc_data = data[data!=1.0]
    prob = np.sum(data == 1.0)/float(data.shape[0])

    ll_fisk_trunc = np.sum(-np.log(fisk.pdf(trunc_data, *fisk_params)))
    #ll_fisk_nontrunc =  np.sum(-np.log(prob * np.ones()))

    ll_ig_trunc = np.sum(-np.log(invgauss.pdf(trunc_data, *ig_params)))
    #ll_ig_nontrunc = np.sum(np.log(prob))

    ll_ln_trunc = np.sum(-np.log(lognorm.pdf(trunc_data, *lognorm_params)))
    #ll_ln_nontrunc = np.sum(np.log(prob))

    ll_exp_trunc = np.sum(-np.log(expon.pdf(trunc_data, *exp_params)))
    #ll_exp_nontrunc = np.sum(np.log(prob))

    ll_gamma_trunc = np.sum(-np.log(gamma.pdf(trunc_data, *gamma_params)))

    return ll_fisk_trunc, ll_ig_trunc, ll_ln_trunc, ll_exp_trunc, ll_gamma_trunc

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


def run_Models_parallel(tviews, ids):
    start_time = time.time()
    args = []

    print 'Got', len(ids), 'viewers';
    for i in range(len(tviews)):
        print(len(tviews[i]))
        if len(tviews[i]) > 100:
            args.append([ids[i], tviews[i]])
    print 'Will fit on', len(args), 'inputs';

    print("Running Fits Now")
    results = Parallel(n_jobs=-1)(map(delayed(fit_models), args))
    print("Time taken to run parallel fits=" + str(time.time() - start_time))
    return results

def compute_BIC(data, n_params, neg_ll):
    bic = (np.log(data.shape[0]) * n_params) - 2 * -neg_ll
    return bic

def fit_models(args):
    viewer_id = args[0]
    viewer_views = np.array(args[1])

    # Making sure we are capping viewer_views at 1.0
    viewer_views[viewer_views >= 1.0] = 1.0;
    #print "Views: ", viewer_views
    # ln_params[0]: prior, ln_params[1]
    #loop_prob, prob_completed, ln_trunc_fit, ln_trueloop_fit
    ln_params = fit_lognormal_loop(viewer_views)
    lognorm_nll = ln_params[3]
    ln_rvs = lognorm_rvs(ln_params, viewer_views.shape[0])
    ln_ks = ks_2samp(ln_rvs, viewer_views)
    ln_bic = compute_BIC(viewer_views, 6, lognorm_nll)
    ln_model_res = [viewer_id, viewer_views.shape[0], lognorm_nll, ln_ks, ln_params, ln_bic]


    fisk_params = fit_fisk_loop(viewer_views)
    fisk_nll = fisk_params[3]
    rvs_fisk = fisk_rvs(fisk_params, viewer_views.shape[0])
    fisk_ks = ks_2samp(rvs_fisk, viewer_views)
    fisk_bic = compute_BIC(viewer_views, 6, fisk_nll)
    fisk_model_res = [viewer_id, viewer_views.shape[0], fisk_nll, fisk_ks, fisk_params, fisk_bic]


    ig_params = fit_ig_loop(viewer_views)
    ig_nll = ig_params[3]
    rvs_ig = ig_rvs(ig_params, viewer_views.shape[0])
    ig_ks = ks_2samp(rvs_ig, viewer_views)
    ig_bic = compute_BIC(viewer_views, 6, ig_nll)
    ig_model_res = [viewer_id, viewer_views.shape[0], ig_nll, ig_ks, ig_params, ig_bic]


    wb_params = fit_weibull_loop(viewer_views)
    wb_nll = wb_params[3]
    rvs_wb = weibull_rvs(wb_params, viewer_views.shape[0])
    wb_ks = ks_2samp(rvs_wb, viewer_views)
    wb_bic = compute_BIC(viewer_views, 6, wb_nll)
    wb_model_res = [viewer_id, viewer_views.shape[0], wb_nll, wb_ks, wb_params, wb_bic]


    gamma_params = fit_gamma_loop(viewer_views)
    gamma_nll = gamma_params[3]
    rvs_gamma = gamma_rvs(gamma_params, viewer_views.shape[0])
    gamma_ks = ks_2samp(rvs_gamma, viewer_views)
    gamma_bic = compute_BIC(viewer_views, 6, gamma_nll)
    gamma_model_res = [viewer_id, viewer_views.shape[0], gamma_nll, gamma_ks, gamma_params, gamma_bic]

    return [ln_model_res, fisk_model_res, ig_model_res, wb_model_res, gamma_model_res]

'''
def process_content(content):
    content = content.decode('utf-8')
    lines = content.split("\n")
    tviews = []
    ids = []

    for line_index in range(1,len(lines)):
        line = lines[line_index]
        line = line.replace("\"","")
        split = line.split(",")

        uid = split[0]
        try:
            vtimes_split = [float(split[x]) for x in range(1,len(split))]
            tviews.append(vtimes_split)
            ids.append(uid)
        except:
            pass

    results = run_Models_parallel(tviews, ids)
    return results
'''

def read_file(in_file):
    f = open(in_file, 'r')

    ids = []
    tviews = []

    line =  f.readline()
    while line:
        line = line.replace('\n', '')
        split = line.split('\t')
        ids.append(split[0])
        split = split[1].split(",")
        tv_arr = []
        for i in range(len(split)):
            tv_arr.append(float(split[i]))
        tv_arr = np.array(tv_arr)
        tviews.append(tv_arr)
        line = f.readline()
    f.close()

    ids = np.array(ids)
    tviews = np.array(tviews)

    return tviews, ids

def overall_proces(inp_file, out_dir):
    out_file = os.path.join(out_dir, "viewers.pkl")
    tviews, ids = read_file(inp_file)
    results = run_Models_parallel(tviews, ids)

    with open(out_file,'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    inp_file = "DATA/Viewers.txt"
    OUT_DIR = "RESULTS_SAMPLE"
    try:
        os.mkdir(os.path.join(OUT_DIR, "Viewers"))
    except:
        pass

    OUT_DIR = os.path.join(OUT_DIR, "Viewers")
    overall_proces(inp_file, OUT_DIR)
