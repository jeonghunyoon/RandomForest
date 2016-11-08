# -*- coding: utf-8 -*-

__author__ = 'Jeonghun Yoon'
'''
RandomForest
uci repository를 이용하여, RandomForest를 실험해본다.
기존의 random forest의 논문과는 다음의 내용이 다르다.
1. 기존 논문에서는 각 노드의 분기에서(base learner를 Regression tree) candidate features를 계속 변화 시켰으나,
여기서는 candidate features를 1개의 base learner에 대해서 고정시키도록 한다.
2. train set에서 랜덤하게 50%의 샘플을 추출하여 learning을 시킨다.
'''

import urllib
import random

from sklearn import tree

##### 1. uci repository에서 wine data를 받아온다.
xData = []
yData = []
f = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
lines = f.readlines()

# column name은 따로 저장한다.
columns = lines.pop(0)

# feature vectors와 label을 나눈다.
for line in lines:
    tokens = line.strip().split(';')
    yData.append(tokens[-1])
    del(tokens[-1])
    xData.append(tokens)


#####  2. train data와 test data를 나눈다.
random.seed(1)
nData = len(xData)
nTest = int(nData * 0.3)
nTrain = nData - nTest

testIdx = random.sample(range(nData), nTest)
testIdx.sort()
testX = [xData[idx] for idx in testIdx]
testY = [yData[idx] for idx in testIdx]
trainX = [xData[idx] for idx in range(nData) if idx not in testIdx]
trainY = [yData[idx] for idx in range(nData) if idx not in testIdx]

##### 3. learning을 수행한다. features는 각 learner마다 다르게 선택한다. regression tree의 경우 논문에서는
#####    전체 feature set의 1/3을 추천했다.
maxTree = 30
treeDepth = 5
nFeat = len(trainX[0])
# candidate features의 수는 전체 features의 1/3으로 한다.
nCanFeat = int(nFeat * 1/3)

# model list와 predict list를 생성한다. predict list에는 n까지의 base learner의 predict 값이 들어간다. 따라서 nCol = 30
modelList = []
predList = []

for iTree in range(maxTree):
    # candidate features를 랜덤하게 추출 (사용자가 nCanFeat의 수를 조정할 수 있다. 하지만 최적값은)
    canFeatIdx = random.sample(range(nFeat), nCanFeat)
    canFeatIdx.sort()

    # train data의 50%를 랜덤하게 추출 (사용자가 추출 비율값을 변경할 수 있다.) candidate features만 가지는 데이터.
    # TODO : 나중에는 xTrain을 대상으로 해보자.
    canTrainIdx = random.sample(range(nTrain), int(nTrain*0.5))
    canTrainX = []
    canTrainY = []
    for idx in canTrainIdx:
        canTrainX.append([trainX[idx][feat] for feat in canFeatIdx])
        canTrainY.append(trainY[idx])

    # model learning
    model = tree.DecisionTreeRegressor(max_depth=treeDepth)
    model.fit(canTrainX, canTrainY)

    # test set을 predict 한 값을 predict list에 넣는다.


##### 4. test를 진행해보고 mse 및 coefficient를 구해본다.


##### 5. plot으로 찍어본다.