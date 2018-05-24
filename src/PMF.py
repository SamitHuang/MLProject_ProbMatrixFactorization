#!/usr/bin/env python
# coding=utf8
'''*************************************************************************
    > File Name: PMF.py
    > Author: HUANG Yongxiang
    > Mail:
    > Created Time: Tue May  8 14:42:13 2018
    > Usage:
*************************************************************************'''

import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import auc, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV
from datetime import datetime

#step size i.e. learning rate
lr = 0.001

# read system args
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--path', type=str, default = "./data.npz")
parser.add_argument('--tune', type=int, default = 1)
parser.add_argument('--max_iter', type=int, default = 100)
parser.add_argument('--verbose', type=int, default = 1)
args = parser.parse_args()

# prepare dataset
path = args.path#"./data.npz"
npzfile = np.load(path )
dataAll = np.asarray([npzfile['user_id'].reshape(-1), npzfile['item_id'].reshape(-1),npzfile['rating'].reshape(-1)] )
dataAll=dataAll.T
[USER_ID_MAX, ITEM_ID_MAX, _] = np.max(dataAll, axis=0) #TODO: n unique

# note that idx starts from 0. user_id=1 ->  row 0 in the matrix
def dataToMatrix(data, dim=[USER_ID_MAX, ITEM_ID_MAX]):
    matrix = np.zeros(dim)
    for ins in data:
        matrix[ins[0]-1,ins[1]-1] = ins[2]
    return matrix

# calc RMSE between given test data and U'V
# rawData -input data with format [[userId, itemId, rating]xN]
def PMF_test(rawData, U, V, useSparse=True):
    if(useSparse):
        mseSum = 0
        for (uID, iID, rij) in rawData:
            i = uID - 1
            j = iID - 1
            mseSum += (rij - U.T[i,:]@V[:,j])**2
        rmse = np.sqrt(mseSum/len(rawData))
    else:
        R_test = dataToMatrix(rawData)
        indicatoR_test = (R_test>0).astype(int)
        R_pred = U.T @ V
        diff = indicatoR_test * (R_test - R_pred)
        rmse = np.sqrt( (diff**2).sum()/indicatoR_test.sum())

    return rmse

#training
#rawData -input data with raw format [[userId, itemId, rating]xN]
def PMF_train(rawData, maxIter=200, K=2,  lamU=0.1, lamV=0.1, useSparse=False ,verbose=False):

    #initialize params.
    np.random.seed(seed=1)
    U= np.random.random_sample((K, USER_ID_MAX))
    V= np.random.random_sample((K, ITEM_ID_MAX))
    n_iter = 0

    if(not useSparse):
        # Matrix Operations
        R_train = dataToMatrix(rawData)
        indicator = (R_train > 0).astype(int)

        R_pred = U.T @ V # matmul, TODO: logistic function to overcome rating out range
        diff = indicator * (R_train - R_pred)

        rmseTrain=[]
        for epoch in range(maxIter):
             #TODO: Linear with # of rating. take advantage of the sparsity of the matrix.
            dEdU = -V @ diff.T  + lamU * U
            dEdV = -U @ diff  + lamV * V
            # sync update
            Ut = U - lr * dEdU #TODO:  - or +
            Vt = V - lr * dEdV
            U = Ut
            V = Vt

            R_pred = U.T @ V #matmul, TODO: logistic function to overcome rating out range
            diff = indicator * (R_train - R_pred)

            n_iter +=1

            rmseTrain.append(np.sqrt( (diff**2).sum()/indicator.sum()))
            # evaluate RMSE loss
            if(verbose):
                print("Epoch {}: Train RMSE: {:.4f}".format(epoch, rmseTrain[epoch]))
    else:
        #Runing Time is Linear to observed rating
        rmseTrain = np.zeros(maxIter)
        for epoch in range(maxIter):
            Ut = U
            Vt = V
            for (uID,iID,rij) in trainData:
                i = uID - 1
                j = iID - 1
                #partially update the latent features of user i and item j
                diff_ij = rij - (U.T)[i,:] @ V[:,j]
                Ut[:,i] += - lr * (-V[:,j] * diff_ij)
                Vt[:,j] += - lr * (-U[:,i] * diff_ij)
                rmseTrain[epoch] += diff_ij**2
            #update U and V
            Ut = Ut - lr * ( lamU * U)
            Vt = Vt - lr * (lamV * V)
            U = Ut
            V = Vt

            rmseTrain[epoch] = np.sqrt(rmseTrain[epoch]/len(trainData))
            n_iter+=1
            if(verbose):
                print("Epoch {}: Train RMSE: {:.4f}".format(epoch, rmseTrain[epoch]))

    return U,V, n_iter, rmseTrain[n_iter-1]

class PMF(BaseEstimator,TransformerMixin):
    def __init__(self, maxIter=200, K=2, lamU=0.1, lamV=0.1 ):
        self.maxIter = maxIter
        self.K = K
        self.lamU = lamU
        self.lamV = lamV

        self.U = np.random.random_sample((K, USER_ID_MAX))
        self.V = np.random.random_sample((K, ITEM_ID_MAX))


    # interface for estimator
    def fit(self, X, y=None, **params):
        U, V, n_iter_, train_rmse_ = PMF_train(X, maxIter=self.maxIter, K=self.K, lamU=self.lamU, lamV=self.lamV)

        #parameters with trailing _ is used to check if the estimator has been fitted
        self.rmse_=  train_rmse_
        self.n_iter_ = n_iter_
        self.U = U
        self.V = V

        return self

    # interface for Grid Search
    def score(self, X, y=None):
        rmse = PMF_test(X, self.U, self.V)
        #print("Scoring: K={}, lamda U={}, lamda V={}, rmse={}".format(self.K,self.lamU, self.lamV, rmse) )
        #since build-in gridsearch pick params by "the bigger the better"
        return -rmse

def GridSearchTuning(trainData, maxIter=100, verbose=0):
    #Tune regularization hyper-params
    regu_params = {"lamU":[0.1,1,10,100],"lamV":[0.1,1,10,100]}
    pmfEst = PMF(K=2, maxIter=maxIter)
    #splitting data into train/validation set by 5-fold
    gs=GridSearchCV(pmfEst, regu_params, cv=5, refit=True, verbose=verbose)
    gs.fit(trainData)

    bestLamU = gs.best_params_['lamU']
    bestLamV = gs.best_params_['lamV']
    #Mean cross-validated score of the best_estimator. (refit on whole training set)
    bestScoreL = -gs.best_score_
    print("\r\nFinish tuning lamda U and lamda V \r\n==> Best lamU: {}, best lamV: {},   Best RMSE={:.4f} ".format(bestLamU, bestLamV,bestScoreL ))
    if(verbose):
        print("Mean test RMSE over 5 fold CV for params (lamda U, lamda V):\r\n{}\r\n".format(-gs.cv_results_["mean_test_score"]))


    #Tune # latent features
    factors_params = {"K":[1,2,3,4,5]}
    pmfEst2 = PMF(lamU=bestLamU,lamV=bestLamV, maxIter=maxIter)
    gs2 =GridSearchCV(pmfEst, factors_params, cv=5, refit=True, verbose=verbose)
    gs2.fit(trainData)

    bestK = gs2.best_params_['K']
    bestScoreK = -gs2.best_score_
    print("Finish tuning factors K \r\n==> Best K={}.   Best RMSE={:.4f}".format(bestK,bestScoreK ))

    if(verbose):
        print("Mean test RMSE over 5 fold CV for params (lamda U, lamda V):\r\n{}".format(-gs2.cv_results_["mean_test_score"]))

    return bestLamU, bestLamV, bestK

#experiment on a given train data and test data
def Experiment(trainData, testData, maxIter=100, verbose=0, tune=1):
    if(tune):
        #Grid Search Tuning on training set
        print("Start grid search CV for {} iterations".format(maxIter))
        bestLamU, bestLamV, bestK = GridSearchTuning(trainData, maxIter=maxIter, verbose=verbose)
    else:
        # load tuning results
        bestLamU = 0.1; bestLamV=0.1
        if len(trainData) > len(testData):
            bestK=2
        else:
            bestK=5
    #train with the full training set using the tuned best hyper-params
    print("-- Training on whole training set with optimal hyperparams---")
    print("Optimal hyper-params: lamda_U = {}, lamda_V={}, K={}".format(bestLamU, bestLamV, bestK))
    startTime = datetime.now()
    U,V,_,rmse_train = PMF_train(trainData, maxIter=maxIter, K=bestK, lamU=bestLamU, lamV=bestLamV, verbose=0 )
    timeCost = datetime.now() - startTime
    print("Finish training in {}s. \r\n=>RMSE on training set: {:.4f}".format(timeCost, rmse_train))

    #evaluate on test set
    rmse_test = PMF_test(testData, U,V)
    print("==> RMSE on test set: {:.4f}".format(rmse_test))

    return rmse_test

if __name__ == "__main__":


    trainData, testData = train_test_split(dataAll, test_size=0.2, random_state=0)

    # Dense training data
    print("\r\n----- Experiment on Dense Data -----")
    rmse1 = Experiment(trainData, testData, maxIter=args.max_iter, verbose=args.verbose, tune=args.tune)
    # Sparse training data

    print("\r\n----- Experiment on Sparse Data -----")
    rmse2 = Experiment(testData, trainData, maxIter=args.max_iter, verbose=args.verbose, tune=args.tune)



