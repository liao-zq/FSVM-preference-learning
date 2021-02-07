#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhiqiang Liao
Date: 2021-01-12 12:55:51
LastEditors: Zhiqiang Liao
LastEditTime: 2021-02-03 20:46:22
FilePath: \FSVM\Section3\feature_engineering.py
Github: https://github.com/liao-zq
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('../FSVM/datasets/comparison/auto-mpg.csv')
# print (lo.shape)

## drop invalid character '?'
drop_Idx = set(df[(df['horsepower'] == '?')].index)
new_Idx = list(set(df.index) - set(drop_Idx))
df = df.iloc[new_Idx]

feature_set = ['cylinders','displacement','horsepower','weight','acceleration','model year','origin']
X = df[feature_set]
df_y = df['mpg']

## convert float data to int
y = df_y.astype('int64')
# print(y)

X=preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf=svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovo')
# clf = LogisticRegression(fit_intercept=True, penalty='l2')
# clf = KNeighborsRegressor(n_neighbors=3)
clf.fit(X_train, y_train)
# X_pred = clf.predict(X_test)
y_pred = clf.predict(X_test)
print(mean_absolute_error(y_test, y_pred))
print (clf.score(X_test, y_test))