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
import numpy as np
import matplotlib.pyplot as plot

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

# testY, trainY를 실수화한다.
trainY = map(float, trainY)
testY = map(float, testY)


##### 3. learning을 수행한다. features는 각 learner마다 다르게 선택한다. regression tree의 경우 논문에서는
#####    전체 feature set의 1/3을 추천했다.
maxTree = 500
treeDepth = 8
nFeat = len(trainX[0])
# candidate features의 수는 전체 features의 1/3으로 한다.
# TODO features의 개수를 3, 4로 했을 때, mse의 차이가 좀 크다.
nCanFeat = int(nFeat * 1/3) + 1

# model list와 predict list를 생성한다. predict list에는 n까지의 base learner의 predict 값이 들어간다. 따라서 nCol = 30
modelList = []
predList = []

for iTree in range(maxTree):
    # candidate features를 랜덤하게 추출 (사용자가 nCanFeat의 수를 조정할 수 있다. 하지만 최적값은)
    canFeatIdx = random.sample(range(nFeat), nCanFeat)
    canFeatIdx.sort()

    # train data의 50%를 랜덤하게 추출 (사용자가 추출 비율값을 변경할 수 있다.) candidate features만 가지는 데이터.
    # 결국 이것은 bagging 기법이라고 볼 수 있다.
    # TODO : 나중에는 xTrain을 대상으로 해보자.
    canTrainIdx = random.sample(range(nTrain), int(nTrain*0.5))
    canTrainIdx.sort()
    canTrainX = []
    canTrainY = []
    for idx in canTrainIdx:
        canTrainX.append([trainX[idx][feat] for feat in canFeatIdx])
        canTrainY.append(trainY[idx])

    # model learning
    model = tree.DecisionTreeRegressor(max_depth=treeDepth)
    model.fit(canTrainX, canTrainY)

    modelList.append(model)

    # test set을 predict 한 값을 predict list에 넣기 전에, test set또한 candidate features만 가지는 데이터로 만든다.
    canTestX = []
    for idx in range(nTest):
        canTestX.append([testX[idx][feat] for feat in canFeatIdx])

    prediction = []
    for testData in canTestX:
        prediction.append(model.predict(testData))

    predList.append(prediction)


##### 4. test를 진행해보고 mse 및 coefficient를 구해본다.
mse = []
coefList = []
allPrediction = []

# n번째 모델까지의 prediction 값의 합을 구한다. 그래야 모델의 수가 증가하면서의 예측값들의 변화량과 mse의 변화량을 알수가 있다.
for iModel in range(maxTree):
    prediction = []
    for idxTest in range(nTest):
        # prediction의 값은 n모델 까지의 prediction의 합의 평균이다.
        prediction.append(sum([predList[idxModel][idxTest] for idxModel in range(iModel+1)]) / (iModel + 1))

    allPrediction.append(prediction)

    # mse를 구한다.
    error = np.array([prediction[i][0] for i in range(nTest)]) - np.array(testY)
    mse.append(sum(error * error) / nTest)

    # coefficient를 구한다.
    coefList.append(np.corrcoef(np.array([prediction[i][0] for i in range(nTest)]), np.array(testY))[0][1])


##### 5. plot으로 찍어본다.
nModels = [i+1 for i in range(len(modelList))]

plot.plot(nModels, mse)
plot.axis('tight')
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
plot.ylim(0.2, max(mse))
plot.show()

plot.figure()

plot.plot(nModels, coefList)
plot.axis('tight')
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Coefficient Correlations')
plot.ylim(min(coefList), 1.0)
plot.show()

print 'Minimum MSE : %f' %(min(mse))
print 'Maximum CorrCoef : %f' %(max(coefList))