#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhiqiang Liao
Date: 2021-02-03 15:38:28
LastEditors: Zhiqiang Liao
LastEditTime: 2021-02-05 16:46:19
FilePath: \FSVM\OR-FSVM.py
Github: https://github.com/liao-zq
'''

from math import e
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error

def CumScore(y_pre, y_test):
    '''
    description: cumulative score(CS) for model selection
    param {v is ndearry type from y_test, y_pre is prediction label}
    return {cumulative score}
    '''
    yi = 0
    e = 0
    # print(y_pre)
    # e = np.arange(1, 11, 1)
    v = y_test.values
    for i in range(1175):
        # print(i)
        # print(y_pre[i])
        # print(v[i])
        if abs(y_pre[i]-v[i])<=e:
            # print(abs(clf_pre[i]-v[i]))
            yi=yi+1
        else:
            yi=yi
    cs = yi/1175*100
    print(yi)

    return cs

def RankFun(X_train, X_test, y_train):
    '''
    description: define multiple hyperplane ranker using svm from sklearn API
    param {input training data and test data}
    return {rank function value}
    '''
    clf_pre = np.zeros(shape=(1175,))
    # y_train = y_train.values
    for i in range(5):
        # print(y_train)
        if i==0:
            y_train1 = y_train.replace(2, 1).replace(3, 1).replace(4, 1).replace(5, 1)
            # print(y_train1)
            clf = svm.SVC()
            clf.fit(X_train, y_train1)
            clf_pre1 = clf.predict(X_test)
            clf_pre = clf_pre+clf_pre1
        elif i==1:
            # print(y_train)
            y_train2 = y_train.replace(1, 0).replace(2, 1).replace(3, 1).replace(4, 1).replace(5, 1)
            clf = svm.SVC()
            clf.fit(X_train, y_train2)
            clf_pre2 = clf.predict(X_test)
            clf_pre = clf_pre+clf_pre2
        elif i==2:
            y_train3 = y_train.replace(1, 0).replace(2, 0).replace(3, 1).replace(4, 1).replace(5, 1)
            clf = svm.SVC()
            clf.fit(X_train, y_train3)
            clf_pre3 = clf.predict(X_test)
            clf_pre = clf_pre+clf_pre3
        elif i==3:
            y_train4 = y_train.replace(1, 0).replace(2, 0).replace(3, 0).replace(4, 1).replace(5, 1)
            clf = svm.SVC()
            clf.fit(X_train, y_train4)
            clf_pre4 = clf.predict(X_test)
            clf_pre = clf_pre+clf_pre4
        elif i==4:
            y_train5 = y_train.replace(1, 0).replace(2, 0).replace(3, 0).replace(4, 0).replace(5, 1)
            clf = svm.SVC()
            clf.fit(X_train, y_train5)
            clf_pre5 = clf.predict(X_test)
            clf_pre = clf_pre+clf_pre5
        else:
            clf_pre = 0
                    

    return clf_pre
    
# upload dataset from UCI
df = pd.read_csv('../FSVM/datasets/Parkinson/data_parkinson_total_score.csv')

# training feature set
feature_set = ['age', 'sex', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 
                'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 
                'NHR', 'HNR', 'RPDE', 'DFA', 'PPE' ]

# split to train and test data
X = df[feature_set]
y = df['total_UPDRS']
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# # Threshold model 1
# from mord import LogisticAT
# logit = LogisticAT()
# logit.fit(X_train, y_train)
# clf_pre = logit.predict(X_test)
# print(mean_absolute_error(y_test, clf_pre))
# print(CumScore(clf_pre, y_test))

# # Threshold model 2
# from mord import LogisticIT
# logit = LogisticIT()
# logit.fit(X_train, y_train)
# clf_pre = logit.predict(X_test)
# print(mean_absolute_error(y_test, clf_pre))
# print(CumScore(clf_pre, y_test))

# # svr
# from sklearn import svm
# clf = svm.SVR()
# clf.fit(X_train, y_train)
# clf_pre = clf.predict(X_test)
# # y_pre = clf_pre.astype('int64')
# print(mean_absolute_error(y_test, clf_pre))
# print(CumScore(clf_pre, y_test))

# # svm
# from sklearn import svm
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# clf_pre = clf.predict(X_test)
# print(clf_pre)
# print(mean_absolute_error(y_test, clf_pre))
# print(CumScore(clf_pre, y_test))

# # Naive bayes
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(X_train, y_train)
# clf_pre = clf.predict(X_test)
# print(mean_absolute_error(y_test, clf_pre))
# print(CumScore(clf_pre, y_test))

# # Logistic regression
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(fit_intercept=True, penalty='l2')
# clf.fit(X_train, y_train)
# clf_pre = clf.predict(X_test)
# print(mean_absolute_error(y_test, clf_pre))
# print(CumScore(clf_pre, y_test))

# # KNN
# from sklearn.neighbors import KNeighborsRegressor
# clf = KNeighborsRegressor(n_neighbors=2)
# clf.fit(X_train, y_train)
# clf_pre = clf.predict(X_test)
# print(mean_absolute_error(y_test, clf_pre))
# print(CumScore(clf_pre, y_test))

# # regression ordianl
# from mord import OrdinalRidge
# clf = OrdinalRidge()
# clf.fit(X_train, y_train)
# clf_pre = clf.predict(X_test)
# print(mean_absolute_error(y_test, clf_pre))
# print(CumScore(clf_pre, y_test))

# MHR (proposed approach)
Rank_score = RankFun(X_train, X_test, y_train)
y_pre = Rank_score.astype('int64')
print(y_pre)
print(mean_absolute_error(y_test, y_pre))
print(CumScore(y_pre, y_test))