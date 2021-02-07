#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhiqiang Liao
Date: 2021-02-06 15:09:26
LastEditors: Zhiqiang Liao
LastEditTime: 2021-02-06 15:43:26
FilePath: \Ranking_SVM-master\r_train.py
Github: https://github.com/liao-zq
'''
from sklearn import svm
from pair import pair

def r_train(x,y):

   x2,y2=pair(x,y)
   print(x2.shape)
   svc=svm.SVC(kernel='linear').fit(x2,y2)

   return svc