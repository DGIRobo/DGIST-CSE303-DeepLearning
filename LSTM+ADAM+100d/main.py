# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:09:53 2023

@author: fist5
"""

import LSTM_Model
import emo_utils
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Train Data Load
x, y = emo_utils.read_csv(r'train_emoji.csv')
XYset = []
for i in range(len(x)):
    reshaped_x = np.reshape(x[i].split()+[0]*(10-len(x[i].split())), (10, 1))
    reshaped_y = np.zeros((5, 1))
    reshaped_y[y[i]] = 1 
    XYset.append([reshaped_x, reshaped_y])

# Test Data Load
x_t, y_t = emo_utils.read_csv(r'test_emoji.csv')
XY_tset = []
for i in range(len(x_t)):
    reshaped_x_t = np.reshape(x_t[i].split()+[0]*(10-len(x_t[i].split())), (10, 1))
    reshaped_y_t = np.zeros((5, 1))
    reshaped_y_t[y_t[i]] = 1 
    XY_tset.append([reshaped_x_t, reshaped_y_t])
    
# Model Generation
Nodes_at_each_layer = 10
learningRate = 1e-3
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
dropout_rate = 0
LSTM = LSTM_Model.LSTM(Nodes_at_each_layer, learningRate, beta1, beta2, epsilon, dropout_rate)

# Traning Parameter Setting
batchSize = 50
maxEpoch = 700
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
    LSTM.activateDropout()
    for step in range(batchSize):
        input = sampling_batch[step][0]
        label = sampling_batch[step][1]
        beforeSoftMax = LSTM.forward(input)
        afterSoftMax = LSTM.SoftMax(beforeSoftMax)
        Loss = LSTM.CrossEntropyLoss(label, afterSoftMax)
        accumulateLoss = accumulateLoss + Loss[0][0]
        output_gradient = LSTM.dotCrossEntropyLoss(label, afterSoftMax)
        LSTM.backward(output_gradient)
    print("Iterations ", iteration, "/", maxIteration, " Average Loss: ", accumulateLoss/batchSize)
    Iteration_data.append(iteration)
    TrainingLoss_data.append(accumulateLoss/batchSize)
    LSTM.update()
    LSTM.deactivateDropout()
    
    # Testing Test Data Set
    testing_batch = random.sample(XY_tset, batchSize)
    for step in range(batchSize):
        testInput = sampling_batch[step][0]
        testLabel = sampling_batch[step][1]
        beforeSoftMax = LSTM.forward(testInput)
        afterSoftMax = LSTM.SoftMax(beforeSoftMax)
        Loss = LSTM.CrossEntropyLoss(testLabel, afterSoftMax)
        accumulateTestLoss = accumulateTestLoss + Loss[0][0]
    TestLoss_data.append(accumulateTestLoss/batchSize)

# Loss Graph Plotting    
plt.plot(Iteration_data, TrainingLoss_data, 'b-', label = 'train')
plt.plot(Iteration_data, TestLoss_data, 'r-', label = 'test')
plt.title('Loss of LSTM Training & Test')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.xlim([0, maxIteration])
plt.ylim([0, 5])
plt.legend(loc='best', ncol=2) 
#plt.show()
plt.savefig('LossGraph.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)

# Test Data Prediction
predictionDatas = []
predictionCM = []
actualCM = []
top3Datas = {0:[[0,0,0], [0,0,0]], 1:[[0,0,0], [0,0,0]], 2:[[0,0,0], [0,0,0]], 3:[[0,0,0], [0,0,0]], 4:[[0,0,0], [0,0,0]]}
for step in range(len(XY_tset)):
    input = XY_tset[step][0]
    label = XY_tset[step][1]
    beforeSoftMax = LSTM.forward(input)
    probability = LSTM.SoftMax(beforeSoftMax)
    predictionDatas.append([input, label, probability])
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
        top3Datas[actualNumber][1][renewIndex] = input
        tempA = np.sort(top3Datas[actualNumber][0])
        tempAIndex = np.argsort(top3Datas[actualNumber][0])
        tempB = [top3Datas[actualNumber][1][i] for i in tempAIndex]
        top3Datas[actualNumber][0] = list(tempA)
        top3Datas[actualNumber][1] = list(tempB)

# Confusion Matrix Plot 
plt.figure()
cm = np.round(confusion_matrix(actualCM, predictionCM, normalize='true'), 2)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of LSTM')
#plt.show()
plt.savefig('ConfusionMatrix.png', dpi=300)

# Prediction Table Plot 
plt.figure()
cm = np.round(confusion_matrix(actualCM, predictionCM), 2)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Prediction Table of LSTM')
#plt.show()
plt.savefig('PredictionTable.png', dpi=300)

# Top 3 Data Plot
plt.figure(figsize=(12,5))
for i in range(5):
    for j in range(3):
        plt.subplot(5, 6, 6*i+j+1)
        string = np.reshape(top3Datas[i][1][2-j], (1, 10))[0].tolist()
        while '0' in string:
            string.remove('0')
        plt.rc('font', size=8)
        plt.text(0.5,0.5,' '.join(string), verticalalignment='center' , horizontalalignment='center' )
        plt.axis('off')
    for j in range(3):
        plt.subplot(5, 6, 6*i+j+4)
        plt.rc('font', size=8)
        plt.text(0.5,0.5,str(round(top3Datas[i][0][2-j]*100, 4))+'%', verticalalignment='center' , horizontalalignment='center' )
        plt.axis('off')
#plt.show()
plt.savefig('Top3Datas.png', dpi=300)