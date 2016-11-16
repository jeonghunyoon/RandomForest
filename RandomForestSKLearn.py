# -*- coding: utf-8 -*-
'''
SKLearn에 있는 RandomForestRegressor를 이용하여 uci archive의 wine 데이터를 분석한다.
'''

import urllib
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot

### 1. uci archive에서 와인 데이터를 불러온다.
f = urllib.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')
lines = f.readlines()
title = lines.pop(0)
title = title.strip().replace('"', '')
titles = np.array(title.split(';'))

xData = []
yData = []
for line in lines:
    tokens = line.strip().split(';')
    yData.append(float(tokens[-1]))
    del (tokens[-1])
    xData.append(map(float, tokens))

xData = np.array(xData)
yData = np.array(yData)


### 2. test set과 train set으로 분리한다. 30%로 분리할 것이고 corss_validation을 사용할 것이다.
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=531)


### 3. RandomForestRegressor를 사용하여 모델을 fit한다.
### 4. mse를 구해본다.
### 5. coeffcient corelation을 구해본다.
mse = []
corrList = []

nTreeList = range(50, 100, 10)
depth = 10
maxFeat = int(xTrain.shape[1] * 0.3)

for nTree in nTreeList:
    model = ensemble.RandomForestRegressor(n_estimators=nTree, max_depth=depth, max_features=maxFeat, oob_score=False,
                                           random_state=531)
    model.fit(xTrain, yTrain)
    prediction = model.predict(xTest)

    # mse를 구해준다.
    mse.append(mean_squared_error(yTest, prediction))

    # correlation coefficient를 구해준다.
    corrList.append(np.corrcoef(yTest, prediction)[0][1])


### 6. features의 중요도를 체크해보고, 모델의 개수에 따른 plot을 실시한다.
plot.plot(nTreeList, mse)
plot.xlabel('Number of Trees in Ensemble.')
plot.ylabel('Mean Squared Error.')
plot.show()

plot.figure()
plot.plot(nTreeList, corrList)
plot.xlabel('Number of Trees in Ensemble.')
plot.ylabel('Correlation Coefficient.')
plot.show()


### 7. 중요한 features를 뽑아보자.
plot.figure()
featureImportance = model.feature_importances_
featureImportance = featureImportance / featureImportance.max()
sortedIdx = np.argsort(featureImportance)

# 히스토그램
barPos = np.arange(sortedIdx.shape[0]) + 0.5
plot.barh(barPos, featureImportance[sortedIdx], align='center')
plot.yticks(barPos, titles[sortedIdx])
plot.xlabel('Variable Importance')
plot.show()