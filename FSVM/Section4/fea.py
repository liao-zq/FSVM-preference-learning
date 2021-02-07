#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhiqiang Liao
Date: 2021-01-28 16:43:57
LastEditors: Zhiqiang Liao
LastEditTime: 2021-01-31 20:30:20
FilePath: \FSVM\Section4\fea.py
Github: https://github.com/liao-zq
'''
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import preprocessing
import seaborn as sns

df = pd.read_excel('../FSVM/section4/parkinsons_updrs.xlsx')
# print(df.shape)

feature_set = ['age','sex','test_time','Jitter(%)','Jitter(Abs)','Jitter:RAP',
                'Jitter:PPQ5','Jitter:DDP','Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5',
                'Shimmer:APQ11','Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE']


X = df[feature_set]
y = df['total_UPDRS']
print(y.dtypes)
X = preprocessing.scale(X)

# convert float data to int
# for i in y:
#     i = int(i)
#     y.append(i)
# print(y.shape)

# print(X.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Log regression
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score

# clf = LogisticRegression(fit_intercept=True, penalty='l2')

# # # svm
# # from sklearn import svm
# # clf = svm.SVC()

# clf.fit(X_train, y_train)
# clf_pred = clf.predict(X_test)
# pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(clf_pred, name='Predict'), margins=True)
# print("Accuracy is {0:.2f}".format(accuracy_score(y_test, clf_pred)))
# print("Precision is {0:.2f}".format(precision_score(y_test, clf_pred, average='macro')))
# print("Recall is {0:.2f}".format(recall_score(y_test, clf_pred, average='macro')))