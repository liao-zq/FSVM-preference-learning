#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhiqiang Liao
Date: 2021-01-06 10:30:03
LastEditors: Zhiqiang Liao
LastEditTime: 2021-02-06 23:12:52
FilePath: \FSVM\Section3\main_RSVM.py
Github: https://github.com/liao-zq
'''

import numpy as np
import csv
import sys
sys.path.append("../FSVM/lib")
from baseline import baseline
from r_train import r_train
from r_predict import r_predict
from pca import pca

# data
train=np.arange(92)
test=np.arange(88)

x=[]
for t in csv.reader(open('../FSVM/datasets/Rank/x.csv', 'r')):
   x.append(t)
x=np.asarray(x,'f')
# x=pca(x)

y=[]
for t in csv.reader(open('../FSVM/datasets/Rank/y.csv', 'r')):
   y.append(t)
y=np.asarray(y,'d')


# train
rsvm=r_train(x[train],y[train])
# rank
r=r_predict(rsvm,x[test])
#r=baseline(x[test],x[-4:])

import pylab as pl
pl.scatter(r[:,0],r[:,1])
pl.plot([0,len(r)],[r[4,1],r[4,1]],'k--',lw=2)
pl.xlabel('CANDIDATE_ID')
pl.ylabel('SCORE')
pl.show()
