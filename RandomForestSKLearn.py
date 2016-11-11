# -*- coding: utf-8 -*-
'''
SKLearn에 있는 RandomForestRegressor를 이용하여 uci archive의 wine 데이터를 분석한다.
'''

import urllib
import numpy as np
from sklearn.cross_validation import train_test_split

### 1. uci archive에서 와인 데이터를 불러온다.
f = urllib.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')
lines = f.readlines()
title = lines.pop(0)

xData = []
yData = []
for line in lines:
    tokens = line.strip().split(';')
    yData.append(float(tokens[-1]))
    del(tokens[-1])
    xData.append(map(float, tokens))

xData = np.array(xData)
yData = np.array(yData)

### 2. test set과 train set으로 분리한다. 30%로 분리할 것이고 corss_validation을 사용할 것이다.
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=531)


### 3. RandomForestRegressor를 사용하여 모델을 fit한다.


### 4. mse를 구해본다.


### 5. coeffcient corelation을 구해본다.


### 6. features의 중요도를 체크해보고, 모델의 개수에 따른 plot을 실시한다.