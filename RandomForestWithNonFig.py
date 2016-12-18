# -*- coding: utf-8 -*-
### RandomForest with non-figure feature dataset
### Uci-archieve : http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data

__author__ = "Jeonghun Yoon"
import urllib2
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plot
import numpy as np


### 1. Load dataset.
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
dataset = urllib2.urlopen(url)

xData = []
yData = []
for row in dataset:
    data = row.strip().split(",")
    # xData extraction
    xData.append(data[:-1])
    # label extraction
    yData.append(float(data[-1]))


### 2. Non-figure features
xDataUpdated = []
for data in xData:
    genFear = [0.0, 0.0]
    if data[0] == "M": genFear[0] = 1.0
    if data[1] == "F": genFear[1] = 1.0
    # Delete non figure feature.
    del (data[0])
    xDataUpdated.append(genFear + data)

titles = np.array(["Sex1", "Sex2", "Length", "Diameter", "Height", "Whole weight", "Shunked weight", "Viscera weight",
          "Shell weight", "Rings"])

nData = len(xDataUpdated)
nCol = len(xDataUpdated[0])


### 3. Divide train set and
xTrain, xTest, yTrain, yTest = train_test_split(xDataUpdated, yData, test_size=0.30, random_state=531)


### 4. Model fitting and MSE check (to test set for performance of model)
# Number of tree used in ensemble model
nTreeList = range(50, 500, 10)
mse = []
for nTree in nTreeList:
    depth = None
    maxFeat = int(nCol * 0.5)
    rfModel = RandomForestRegressor(n_estimators=nTree, max_depth=depth, max_features=maxFeat, oob_score=False,
                                    random_state=531)
    rfModel.fit(xTrain, yTrain)
    prediction = rfModel.predict(xTest)
    mse.append(mean_squared_error(yTest, prediction))

print "MSE"
print mse[-1]


### 5. Plot
plot.figure()
plot.plot(nTreeList, mse)
plot.xlabel("Number of trees in ensemble")
plot.ylabel("Mean Squared Error")
# plot.ylim([0.0, max(mse)])
plot.show()


### 6. Importance of features
featureImportance = rfModel.feature_importances_
featureImportance = featureImportance / featureImportance.max()
sortedIdx = np.argsort(featureImportance)
barPos = np.arange(sortedIdx.shape[0]) + .5
plot.figure()
plot.barh(barPos, featureImportance[sortedIdx], align = "center")
plot.yticks(barPos, titles[sortedIdx])
plot.xlabel("Variable Importance")
plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plot.show()