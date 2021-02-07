#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhiqiang Liao
Date: 2021-02-03 17:18:12
LastEditors: Zhiqiang Liao
LastEditTime: 2021-02-05 17:31:11
FilePath: \FSVM\fig4.8.py
Github: https://github.com/liao-zq
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def CumArray(y_pre, y_test):
    '''
    description: cumulative score(CS) for model selection
    param {v is ndearry type from y_test, y_pre is prediction label}
    return {cumulative score}
    '''
    yi = 0
    yj = []
    e = np.arange(0, 10, 1)
    v = y_test.values
    for j in range(10):
        yi = 0
        for i in range(1175):
            if abs(y_pre[i]-v[i])<=e[j]:
                # print(y_pre[i])
                # print(v[i])
                # print(e[j])
                yi=yi+1
            else:
                yi=yi
        yj.append(yi)  
    yj = np.array(yj)
    # print(yj)

    cs = yj/1175*100
    # print(cs)
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

# svc
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
clf_pre_svm = clf.predict(X_test)

# svr
from sklearn import svm
clf = svm.SVR()
clf.fit(X_train, y_train)
clf_pre_svr = clf.predict(X_test)

# Threshold model
from mord import LogisticAT
logit = LogisticAT()
logit.fit(X_train, y_train)
clf_pre_LogisticAT = logit.predict(X_test)

# Threshold model
from mord import LogisticIT
logit = LogisticIT()
logit.fit(X_train, y_train)
clf_pre_LogisticIT = logit.predict(X_test)

# regression ordianl
from mord import OrdinalRidge
clf = OrdinalRidge()
clf.fit(X_train, y_train)
clf_pre_OrdinalRidge = clf.predict(X_test)

# OR-FSVM (proposed approach)
Rank_score = RankFun(X_train, X_test, y_train)

# 1
y_svm = CumArray(clf_pre_svm, y_test)
#2
y_svr = CumArray(clf_pre_svr, y_test)
#3
y_LogisticAT= CumArray(clf_pre_LogisticAT, y_test)
#4
y_LogisticIT = CumArray(clf_pre_LogisticIT, y_test)
#5
y_OrdinalRidge = CumArray(clf_pre_OrdinalRidge, y_test)
#6
y_FSVM = CumArray(Rank_score, y_test)


x_svm = np.arange(0, 10, 1)
plt.plot(x_svm, y_svm, label='SVM',linewidth=1,marker='.')
plt.plot(x_svm, y_svr, label='SVR',linewidth=1,marker='x')
plt.plot(x_svm, y_LogisticAT, label='LogisticAT',linewidth=1,marker='*')
plt.plot(x_svm, y_LogisticIT, label='LogisticIT',linewidth=1,marker='v')
plt.plot(x_svm, y_OrdinalRidge, label='Logistic',linewidth=1,marker='|')
plt.plot(x_svm, y_FSVM, label='OR-FSVM',linewidth=1,marker='d')
plt.xlabel('Error Level $\\itT$')
plt.ylabel('Cumulation Score (%)')
plt.legend()
plt.show()