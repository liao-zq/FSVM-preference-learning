#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhiqiang Liao
Date: 2021-01-06 13:21:17
LastEditors: Zhiqiang Liao
LastEditTime: 2021-01-26 20:51:58
FilePath: \coding\Section3\test.py
Github: https://github.com/liao-zq
'''

import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

k = np.arange(0,12)
cost = abs(4-k)


    
plt.plot(k, cost)

plt.xlim((0, 10))
plt.ylim((0,7))
plt.xlabel('class', fontsize=11)
plt.ylabel('loss', fontsize=11)
plt.grid()
plt.show()

# print(Y1[1:100]) 