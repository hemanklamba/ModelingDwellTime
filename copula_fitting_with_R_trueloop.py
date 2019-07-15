import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from datetime import datetime
from scipy import stats
import scipy
from scipy.stats import rankdata, fisk, expon, pareto, lognorm, weibull_min
from progressbar import ProgressBar
from fitter import Fitter
import pickle
import matplotlib.pyplot as plt

def get_params_NEW(BASE_DIR, params_file):
    tview_loopTrue_params = pickle.load(open(os.path.join(BASE_DIR, params_file), "r"))
    story_ids = []
    nViews = []
    fisk_params = []
    pbar = ProgressBar()

    for i in pbar(range(len(tview_loopTrue_params))):
        story_ids.append(tview_loopTrue_params[i][0])
        nViews.append(tview_loopTrue_params[i][1])
        fisk_params.append(
            [tview_loopTrue_params[i][2][15][0], tview_loopTrue_params[i][2][15][2]])

    fisk_params = np.array(fisk_params)
    return fisk_params, story_ids, nViews

def create_pairplot(df, BASE_DIR, fname):
    matrixPairPlot(df, savefig=os.path.join(BASE_DIR, fname))

def create_train_test(df_false, num_train):
    sampled_idxs = np.random.choice(range(len(df_false)), size=num_train, replace=False)
    #nonsampled_idxs = [(x not in sampled_idxs) for x in range(len(df_false))]
    #print len(sampled_idxs), len(nonsampled_idxs), np.sum(nonsampled_idxs)
    #test_df = df_false.iloc[nonsampled_idxs]
    test_df = df_false.copy()
    train_df = df_false.iloc[sampled_idxs]

    return train_df, test_df

def fitting_marginals(inp_data):
    # inp_data: Input is a dataframe
    # Returns:
    fits = [];
    for i in xrange(inp_data.shape[1]):
        f = Fitter(inp_data.iloc[:, i]);
        f.fit();

        # choose best distribution and best parameter fit
        j = 0
        found = False
        while (found == False):
            cand_dist = f.df_errors.sort_values('sumsquare_error').iloc[j].name

            if cand_dist in f.fitted_param.keys():
                best_dist = cand_dist
                best_params = f.fitted_param[cand_dist]
                found = True
            j += 1
        fits.append((best_dist, best_params));
        f.summary()

    # generate scipy rv objects for marginals
    marginal_fits = [eval('scipy.stats.' + fits[i][0])(*fits[i][1]) for i in xrange(len(fits))]
    return marginal_fits

def convert_inp_data(train_df, marginal_fits, BASE_DIR, marginal_fname):
    conv_inp_data = train_df.copy()

    for i in xrange(train_df.shape[1]):
        conv_inp_data.iloc[:, i] = marginal_fits[i].cdf(train_df.iloc[:, i])
        assert min(conv_inp_data.iloc[:, i]) >= 0.0
        assert max(conv_inp_data.iloc[:, i]) <= 1.0
        plt.figure()
        plt.hist(conv_inp_data.iloc[:, i])
        plt.savefig(os.path.join(BASE_DIR, marginal_fname+str(i)+".pdf"))
        plt.close()

    return conv_inp_data

def generate_Anom_data(anom_mode, num_noise_samples):
    anom_views = []

    if anom_mode == 1:
        for i in range(num_noise_samples):
            nViews = int(np.random.uniform(low=100, high=1000))
            falseEntries = np.ones((nViews,))
            anom_views.append(falseEntries)

    elif anom_mode == 2:
        for i in range(num_noise_samples):
            nViews = int(np.random.uniform(low=100, high=1000))
            complete_prob = np.random.uniform(low=0.8, high=0.9)
            falseEntries = np.random.uniform(low=0.8, high=0.9, size=nViews)
            compl_indexes = np.random.choice(range(falseEntries.shape[0]), size=int(complete_prob * nViews))
            falseEntries[compl_indexes] = 1.0
            anom_views.append(falseEntries)

    elif anom_mode == 3:
        ## EXPON
        for i in range(num_noise_samples):
            nViews = int(np.random.uniform(low=100, high=1000))
            views = expon.rvs(loc=0.01, scale=0.01, size=nViews)
            views[views>1.0] = 1.0
            anom_views.append(views)

    elif anom_mode == 4:
        ## Pareto 5.0 Loc=0.5
        ## Pareto 20.0 Loc=0.0
        for i in range(num_noise_samples):
            nViews = int(np.random.uniform(low=100, high=1000))
            #views = pareto.rvs(5.0, loc=0.5, size=nViews) - 1.0
            views = pareto.rvs(20.0, size=nViews) - 1.0
            views[views>1.0] = 1.0
            anom_views.append(views)

    elif anom_mode == 5:
        ## LogNorm
        for i in range(num_noise_samples):
            nViews = int(np.random.uniform(low=100, high=1000))
            views = lognorm.rvs(s=1, scale=0.01, size=nViews)
            views[views>1.0] = 1.0
            anom_views.append(views)

    elif anom_mode == 6:
        ## Weibull_Min
        for i in range(num_noise_samples):
            nViews = int(np.random.uniform(low=100, high=1000))
            views = weibull_min.rvs(0.05, size=nViews)
            views[views>1.0] = 1.0
            anom_views.append(views)

    anom_views = np.array(anom_views)

    pbar = ProgressBar()
    synth_fisk_params = []
    for i in pbar(range(len(anom_views))):
        synth_single = fit_fisk(anom_views[i], 0)
        synth_fisk_params.append([synth_single[0], synth_single[1][0], synth_single[1][2]])

    print len(synth_fisk_params)
    synth_fisk_params = np.array(synth_fisk_params)
    return synth_fisk_params

def get_files_for_copula_fitting(BASE_DIR, percent=None):
    params_file = "../Sep2021_IndividualModels_100_LoopFalseFits_Loc0.pkl"
    marginal_fname = "FittedMarginal_"+str(percent)+"_"

    fisk_params, story_ids, nViews = get_params(BASE_DIR, params_file)

    df_false = pd.DataFrame(fisk_params)
    if percent==None:
        num_train = 20000
    else:
        num_train = int((percent/100.0) * df_false.shape[0])

    train_df, test_df = create_train_test(df_false, num_train)

    print "Training Shape="+str(train_df.shape)
    print "Testing Shape=" + str(test_df.shape)

    marginal_fits = fitting_marginals(train_df)
    conv_inp_data = convert_inp_data(train_df, marginal_fits, BASE_DIR, marginal_fname)

    orig_data = np.vstack((train_df.iloc[:,0], train_df.iloc[:,1], train_df.iloc[:,2])).T

    trainingPDFs = []
    for i in range(orig_data.shape[0]):
        inp_data = orig_data[i, :]
        pdfs = np.array([marginal_fits[j].pdf(inp_data[j]) for j in range(len(inp_data))])
        trainingPDFs.append(pdfs)
    trainingPDFs = np.array(trainingPDFs)

    trainingData = np.vstack((conv_inp_data.iloc[:, 0], conv_inp_data.iloc[:, 1], conv_inp_data.iloc[:, 2])).T

    return trainingData, trainingPDFs, marginal_fits, test_df, story_ids, nViews

def impute_and_marginal_computation1(test_params, train_params, train_marginal_fits):
    num_nans_training = np.array([np.sum(np.isnan(train_params[:, j])) / float(train_params.shape[0])
                                  for j in range(train_params.shape[1])])

    # Now compute the cdf_points
    cdf_points = np.array([0.5 - ((1 - num_nans_training[j]) / 2) for j in range(num_nans_training.shape[0])])

    cdf_points = np.minimum(cdf_points, 1 - (1e-20))
    cdf_points = np.maximum(cdf_points, 1e-20)

    # Now get points which have cdf as that cdf_points
    data_points = np.array([train_marginal_fits[j].ppf(cdf_points[j]) for j in range(cdf_points.shape[0])])

    # Getting PDF of the points
    pdf_points = np.array([train_marginal_fits[j].pdf(data_points[j]) for j in range(data_points.shape[0])])

    print "Value to Impute CDF with=" + str(cdf_points)
    print "Data points corresponding to CDF as above=" + str(data_points)
    print "Value to Impute PDF with=" + str(pdf_points)

    # Impute PDF values.
    pbar = ProgressBar()
    testPDFs = []
    for i in pbar(range(test_params.shape[0])):
        inp_data = test_params[i, :]
        pdfs = []
        for j in range(len(marginal_fits)):
            if np.isnan(inp_data[j]):
                pdfs.append(pdf_points[j])
            else:
                value = marginal_fits[j].pdf(inp_data[j])
                if np.isnan(value):
                    pdfs.append(pdf_points[j])
                else:
                    pdfs.append(marginal_fits[j].pdf(inp_data[j]))
        testPDFs.append(pdfs)
    testPDFs = np.array(testPDFs)

    print "Got PDF Values"
    pbar = ProgressBar()
    # Get CDF values
    cdf_test = test_params.copy()
    for i in pbar(range(test_params.shape[1])):
        cdf_val = marginal_fits[i].cdf(test_params[:, i])
        cdf_val[np.isnan(cdf_val)] = cdf_points[i]
        cdf_val = np.minimum(cdf_val, 1 - (1e-20))
        cdf_val = np.maximum(cdf_val, 1e-20)
        cdf_test[:, i] = cdf_val

    return cdf_points, pdf_points, cdf_test, testPDFs

def generate_synth_data(anom_mode, noise_percentage):
    if anom_mode == 1:
        n_anom1 = 200
        n_anom2 = 200
        anom_views = []

        for i in range(n_anom1):
            nViews = int(np.random.uniform(low=100, high=1000))
            falseEntries = np.ones((nViews,))
            anom_views.append(falseEntries)

        # Dirac Delta
        for i in range(n_anom2):
            nViews = int(np.random.uniform(low=100, high=1000))
            complete_prob = np.random.uniform(low=0.8, high=0.9)
            falseEntries = np.random.uniform(low=0.8, high=0.9, size=nViews)
            compl_indexes = np.random.choice(range(falseEntries.shape[0]), size=int(complete_prob * nViews))
            falseEntries[compl_indexes] = 1.0
            anom_views.append(falseEntries)

    elif anom_mode == 2:
        ## EXPON Param 0.05
        n_anom1 = 400
        anom_views = []

        for i in range(n_anom1):
            nViews = int(np.random.uniform(low=100, high=1000))
            views = expon.rvs(0.05, size=nViews)
            views[views>1.0] = 1.0
            anom_views.append(views)

    elif anom_mode == 3:
        ##
        n_anom1 = 400
        anom_views = []

        for i in range(n_anom1):
            nViews = int(np.random.uniform(low=100, high=1000))
            views = pareto.rvs(50.0, size=nViews) - 1.0
            views[views>1.0] = 1.0
            anom_views.append(views)

    anom_views = np.array(anom_views)

    pbar = ProgressBar()
    synth_fisk_params = []
    for i in pbar(range(len(anom_views))):
        synth_single = fit_fisk(anom_views[i], 0)
        synth_fisk_params.append([synth_single[0], synth_single[1][0], synth_single[1][2]])

    print len(synth_fisk_params)
    synth_fisk_params = np.array(synth_fisk_params)
    return synth_fisk_params

def fit_fisk(data, mode=1):
    prob = np.sum(data == 1.0) / float(data.shape[0])
    num_one = np.sum(data == 1.0)
    trunc_data = data[data != 1.0]

    fisk_params = [np.nan, 0.0, np.nan]
    if len(trunc_data)>0:
        if mode == 0:
            fisk_params = fisk.fit(trunc_data, floc=0.0)
        else:
            fisk_params = fisk.fit(trunc_data)

    return [prob, fisk_params]

def get_data(BASE_DIR):
    params_file = "LoopTrue.pkl"
    fisk_params, story_ids, nViews = get_params_NEW(BASE_DIR, params_file)
    df_true = pd.DataFrame(fisk_params)
    orig_params = np.vstack((df_true.iloc[:,0], df_true.iloc[:,1])).T
    labels = np.zeros(orig_params.shape[0])

    ov_params = orig_params

    return ov_params, story_ids, nViews, labels

def split_dataset(data, train_percent, labels):
    num_train = int((train_percent/100.0) * data.shape[0])
    sampled_idxs = np.random.choice(range(data.shape[0]), size=num_train, replace=False)
    #nonsampled_idxs = [(x not in sampled_idxs) for x in range(data.shape[0])]
    #print len(sampled_idxs), len(nonsampled_idxs), np.sum(nonsampled_idxs)
    #test_params = data[nonsampled_idxs, :]
    #test_labels = labels[nonsampled_idxs]
    train_params = data[sampled_idxs, :]
    train_labels = labels[sampled_idxs]

    test_params = data.copy()
    test_labels = labels

    return train_params, test_params, train_labels, test_labels

def fit_marginals(train_data):
    fits = [];
    for i in xrange(train_data.shape[1]):
        f = Fitter(train_data[~np.isnan(train_data[:,i]), i]);
        f.fit();

        # choose best distribution and best parameter fit
        j = 0
        found = False
        while (found == False):
            cand_dist = f.df_errors.sort_values('sumsquare_error').iloc[j].name

            if cand_dist in f.fitted_param.keys():
                best_dist = cand_dist
                best_params = f.fitted_param[cand_dist]
                found = True
            j += 1
        fits.append((best_dist, best_params));
        f.summary()

    # generate scipy rv objects for marginals
    marginal_fits = [eval('scipy.stats.' + fits[i][0])(*fits[i][1]) for i in xrange(len(fits))]

    num_nans_training = np.array([np.sum(np.isnan(train_data[:, j])) / float(train_data.shape[0])
                                  for j in range(train_data.shape[1])])

    cdf_points = np.array([0.5 - ((1 - num_nans_training[j]) / 2) for j in range(num_nans_training.shape[0])])

    cdf_points = np.minimum(cdf_points, 1 - (1e-20))
    cdf_points = np.maximum(cdf_points, 1e-20)

    # Now get points which have cdf as that cdf_points
    data_points = np.array([marginal_fits[j].ppf(cdf_points[j]) for j in range(cdf_points.shape[0])])

    # Getting PDF of the points
    pdf_points = np.array([marginal_fits[j].pdf(data_points[j]) for j in range(data_points.shape[0])])

    print "Value to Impute CDF with=" + str(cdf_points)
    print "Data points corresponding to CDF as above=" + str(data_points)
    print "Value to Impute PDF with=" + str(pdf_points)


    # Impute PDF values.
    pbar = ProgressBar()
    train_pdfs = []
    for i in pbar(range(train_data.shape[0])):
        inp_data = train_data[i, :]
        pdfs = []
        for j in range(len(marginal_fits)):
            if np.isnan(inp_data[j]):
                pdfs.append(pdf_points[j])
            else:
                value = marginal_fits[j].pdf(inp_data[j])
                if np.isnan(value):
                    pdfs.append(pdf_points[j])
                else:
                    pdfs.append(marginal_fits[j].pdf(inp_data[j]))
        train_pdfs.append(pdfs)
    train_pdfs = np.array(train_pdfs)

    print "Got PDF Values"
    pbar = ProgressBar()
    # Get CDF values
    cdf_train = train_data.copy()
    for i in pbar(range(train_data.shape[1])):
        cdf_val = marginal_fits[i].cdf(train_data[:, i])
        cdf_val[np.isnan(cdf_val)] = cdf_points[i]
        cdf_val = np.minimum(cdf_val, 1 - (1e-20))
        cdf_val = np.maximum(cdf_val, 1e-20)
        cdf_train[:, i] = cdf_val

    assert min(cdf_train[:, i]) >= 0.0
    assert max(cdf_train[:, i]) <= 1.0

    return marginal_fits, cdf_points, pdf_points, cdf_train, train_pdfs

def impute(test_params, impute_cdf_points, impute_pdf_points, marginals):
    pbar = ProgressBar()
    testPDFs = []
    for i in pbar(range(test_params.shape[0])):
        inp_data = test_params[i, :]
        pdfs = []
        for j in range(len(marginals)):
            if np.isnan(inp_data[j]):
                pdfs.append(impute_pdf_points[j])
            else:
                value = marginals[j].pdf(inp_data[j])
                if np.isnan(value):
                    pdfs.append(impute_pdf_points[j])
                else:
                    pdfs.append(marginals[j].pdf(inp_data[j]))
        testPDFs.append(pdfs)
    testPDFs = np.array(testPDFs)

    print "Got PDF Values"
    pbar = ProgressBar()
    # Get CDF values
    cdf_test = test_params.copy()
    for i in pbar(range(test_params.shape[1])):
        cdf_val = marginals[i].cdf(test_params[:, i])
        cdf_val[np.isnan(cdf_val)] = impute_cdf_points[i]
        cdf_val = np.minimum(cdf_val, 1 - (1e-20))
        cdf_val = np.maximum(cdf_val, 1e-20)
        cdf_test[:, i] = cdf_val


    return cdf_test, testPDFs

if __name__ == '__main__':
    train_percent = int(sys.argv[1])

    BASE_DIR = "RESULTS_SAMPLE/Snaps_LoopTrue"
    marginal_fname = "FittedMarginal_"
    use_synth_data = False

    trainDatafile = "TrainData_" + str(train_percent)+".csv"
    trainProbsfile = "TrainProbs_" + str(train_percent)+".csv"
    trainLabelsfile = "TrainLabels_"+str(train_percent)+".csv"
    testLabelsfile = "TestLabels_"+str(train_percent)+".csv"
    testDatafile = "TestData_" + str(train_percent)+".csv"
    testProbsfile = "TestProbs_" + str(train_percent)+".csv"

    storyIdsfile = "storyIds.pkl"
    nViewsfile = "nviews.csv"

    print "Getting Overall Data"

    ov_params, story_ids, nViews, labels = get_data(BASE_DIR)
    train_params, test_params, train_labels, test_labels = split_dataset(ov_params, train_percent, labels)

    pickle.dump(train_params, open(
        os.path.join(BASE_DIR, 'Orig_TrainData_' + str(train_percent) +'.pkl'), 'w'))
    pickle.dump(test_params, open(
        os.path.join(BASE_DIR, 'Orig_TestData_' + str(train_percent) +'.pkl'), 'w'))

    marginals, impute_cdf_points, impute_pdf_points, train_cdfs, train_pdfs = fit_marginals(train_params)
    test_cdfs, test_pdfs = impute(test_params, impute_cdf_points, impute_pdf_points, marginals)

    '''
    for j in range(train_params.shape[1]):
        plt.figure()
        plt.hist(train_cdfs[:, j])
        plt.savefig(os.path.join(BASE_DIR, marginal_fname+"_"+str(train_percent)+"_"+str(j) + ".pdf"))
        plt.close()
    '''

    pickle.dump(train_cdfs, open(
        os.path.join(BASE_DIR, 'Fitted_TrainData_' + str(train_percent) + '.pkl'), 'w'))
    pickle.dump(marginals, open(
        os.path.join(BASE_DIR, 'Fitted_Marginals_' + str(train_percent) + '.pkl'), 'w'))
    pickle.dump(test_params, open(
        os.path.join(BASE_DIR, 'Fitted_TestData_' + str(train_percent) +  '.pkl'), 'w'))
    pickle.dump(story_ids, open(os.path.join(BASE_DIR, storyIdsfile), 'w'))

    np.savetxt(os.path.join(BASE_DIR, trainDatafile), train_cdfs)
    np.savetxt(os.path.join(BASE_DIR, trainProbsfile), train_pdfs)
    np.savetxt(os.path.join(BASE_DIR, trainLabelsfile), train_labels)
    np.savetxt(os.path.join(BASE_DIR, testLabelsfile), test_labels)
    np.savetxt(os.path.join(BASE_DIR, nViewsfile), nViews)
    np.savetxt(os.path.join(BASE_DIR, testDatafile), test_cdfs)
    np.savetxt(os.path.join(BASE_DIR, testProbsfile), test_pdfs)

    print "==========++TRAINING======================"
    print "Num Nans 0=" + str(np.sum(np.isnan(train_cdfs[:, 0])))
    print "Num Nans 1=" + str(np.sum(np.isnan(train_cdfs[:, 1])))

    assert min(train_pdfs[:, 0]) >= 0.0
    assert min(train_pdfs[:, 1]) >= 0.0

    assert min(train_cdfs[:, 0]) >= 0.0
    assert min(train_cdfs[:, 1]) >= 0.0

    assert max(train_cdfs[:, 0]) <= 1.0
    assert max(train_cdfs[:, 1]) <= 1.0

    print "==========++TESTING======================"
    print "Num Nans 0=" + str(np.sum(np.isnan(test_cdfs[:, 0])))
    print "Num Nans 1=" + str(np.sum(np.isnan(test_cdfs[:, 1])))

    assert min(test_pdfs[:, 0]) >= 0.0
    assert min(test_pdfs[:, 1]) >= 0.0

    assert min(test_cdfs[:, 0]) >= 0.0
    assert min(test_cdfs[:, 1]) >= 0.0

    assert max(test_cdfs[:, 0]) <= 1.0
    assert max(test_cdfs[:, 1]) <= 1.0