# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:35:17 2023

12:20 ~ 

@author: fist5
"""
import CNN
import dataloader
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Train Data Load
train_loader = dataloader.Dataloader(r'C:\Users\user\Desktop\CNN', is_train = True, shuffle = True)
x = train_loader.images / 255
x = np.reshape(x, (len(x), 28*28))
y = train_loader.labels
XYset = []
for i in range(len(x)):
    XYset.append([np.reshape(x[i], (28,28)), np.reshape(y[i], (10, 1))])

# Test Data Load
test_loader = dataloader.Dataloader(r'C:\Users\user\Desktop\CNN', is_train = False, shuffle = False)
x_t = test_loader.images / 255
x_t = np.reshape(x_t, (len(x_t), 28*28))
y_t = test_loader.labels
XY_tset = []
for i in range(len(x_t)):
    XY_tset.append([np.reshape(x_t[i], (28, 28)), np.reshape(y_t[i], (10, 1))])

# Model Generation
FLKernels = 2
FLKernelSize = 5
SLKernels = 2
SLKernelSize = 3
learningRate = 1e-3
CNN = CNN.CNN(28, 28, 1, 10, FLKernelSize, FLKernels, SLKernelSize, SLKernels, learningRate)

# Traning Parameter Setting
batchSize = 100
maxEpoch = 6
maxIteration = int(len(x)/batchSize*maxEpoch)

# For Plot
Iteration_data = []
TrainingLoss_data = []
TestLoss_data = []

# Start Training
for iteration in range(maxIteration):
    accumulateLoss = 0
    accumulateTestLoss = 0
    sampling_batch = random.sample(XYset, batchSize)
    for step in range(batchSize):
        input = [sampling_batch[step][0]]
        label = sampling_batch[step][1]
        beforeSoftMax = CNN.forward(input)
        afterSoftMax = CNN.SoftMax(beforeSoftMax)
        Loss = CNN.CrossEntropyLoss(label, afterSoftMax)
        accumulateLoss = accumulateLoss + Loss[0][0]
        output_gradient = CNN.dotCrossEntropyLoss(label, afterSoftMax)
        CNN.backward(output_gradient)
    print("Iterations ", iteration, "/", maxIteration, " Average Loss: ", accumulateLoss/batchSize)
    Iteration_data.append(iteration)
    TrainingLoss_data.append(accumulateLoss/batchSize)
    CNN.update()
    
    # Testing Test Data Set
    testing_batch = random.sample(XY_tset, batchSize)
    for step in range(batchSize):
        testInput = [sampling_batch[step][0]]
        testLabel = sampling_batch[step][1]
        beforeSoftMax = CNN.forward(testInput)
        afterSoftMax = CNN.SoftMax(beforeSoftMax)
        Loss = CNN.CrossEntropyLoss(testLabel, afterSoftMax)
        accumulateTestLoss = accumulateTestLoss + Loss[0][0]
    TestLoss_data.append(accumulateTestLoss/batchSize)

# Loss Graph Plotting    
plt.plot(Iteration_data, TrainingLoss_data, 'b-', label = 'train')
plt.plot(Iteration_data, TestLoss_data, 'r-', label = 'test')
plt.title('Loss of CNN Training & Test')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.xlim([0, maxIteration])
plt.ylim([0, 2.5])
plt.legend(loc='best', ncol=2) 
#plt.show()
plt.savefig('LossGraph.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)

# Test Data Prediction
predictionDatas = []
predictionCM = []
actualCM = []
top3Datas = {0:[[0,0,0], [0,0,0]], 1:[[0,0,0], [0,0,0]], 2:[[0,0,0], [0,0,0]], 3:[[0,0,0], [0,0,0]], 4:[[0,0,0], [0,0,0]], 5:[[0,0,0], [0,0,0]], 6:[[0,0,0], [0,0,0]], 7:[[0,0,0], [0,0,0]], 8:[[0,0,0], [0,0,0]], 9:[[0,0,0], [0,0,0]]}
for step in range(len(XY_tset)):
    input = [XY_tset[step][0]]
    label = XY_tset[step][1]
    beforeSoftMax = CNN.forward(input)
    probability = CNN.SoftMax(beforeSoftMax)
    predictionDatas.append([np.reshape(input, (28,28)), label, probability])
    actualNumber = list(np.transpose(label)[0]).index(1)
    predictionNumber = list(np.transpose(probability)[0]).index(max(list(np.transpose(probability)[0])))
    #Confusion Matrix Preprocessing
    actualCM.append(actualNumber)
    predictionCM.append(predictionNumber)
    #Top3 Data Preprocessing
    minimumProb = min(top3Datas[actualNumber][0])
    if minimumProb < max(np.transpose(probability)[0]):
        renewIndex = top3Datas[actualNumber][0].index(minimumProb)
        top3Datas[actualNumber][0][renewIndex] = max(np.transpose(probability)[0])
        top3Datas[actualNumber][1][renewIndex] = np.reshape(input, (28,28))
        tempA = np.sort(top3Datas[actualNumber][0])
        tempAIndex = np.argsort(top3Datas[actualNumber][0])
        tempB = [top3Datas[actualNumber][1][i] for i in tempAIndex]
        top3Datas[actualNumber][0] = list(tempA)
        top3Datas[actualNumber][1] = list(tempB)

# Confusion Matrix Plot 
cm = np.round(confusion_matrix(actualCM, predictionCM, normalize='true'), 2)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of CNN')
#plt.show()
plt.savefig('ConfusionMatrix.png', dpi=300)

# Top 3 Data Plot
plt.figure(figsize=(6,10))
for i in range(10):
    for j in range(3):
        img = top3Datas[i][1][2-j]
        plt.subplot(10, 6, 6*i+j+1)
        plt.imshow(img)
        plt.axis('off')
    for j in range(3):
        plt.subplot(10, 6, 6*i+j+4)
        plt.rc('font', size=8)
        plt.text(0.5,0.5,str(round(top3Datas[i][0][2-j]*100, 4))+'%', verticalalignment='center' , horizontalalignment='center' )
        plt.axis('off')
#plt.show()
plt.savefig('Top3Datas.png', dpi=300)
