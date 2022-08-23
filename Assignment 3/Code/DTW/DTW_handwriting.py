import numpy as np
from DTW_Handwritting_Read import *
from DTW import *
import matplotlib.pyplot as plt
import statistics
import seaborn as sns

train = []
dev = []

# Given Datasets
dataset = ['a', 'bA', 'dA', 'lA', 'tA']

# Read Data
for i in dataset:
	train.append(read_data(i, 'train', '.txt'))
	dev.append(read_data(i, 'dev', '.txt'))

# Add feature - Slope between consecutive points

for c in range(5):
  for i in range(len(train[c])):
    data = train[c][i]
    slope = []
    for j in range(len(data)):
      if j == len(data)-1 :
        temp = data[j] - data[j-1]
      else:
        temp = data[j+1]-data[j]
      if temp[0] == 0:
        temp[0] = 10 ** -5
      slope.append(temp[1]/temp[0])
    slope = np.array([slope]).reshape(-1,1)
    train[c][i] = np.hstack((train[c][i],slope))
  for i in range(len(dev[c])):
    data = dev[c][i]
    slope = []
    for j in range(len(data)):
      if j == len(data)-1 :
        temp = data[j] - data[j-1]
      else:
        temp = data[j+1]-data[j]
      if temp[0] == 0:
        temp[0] = 10 ** -5
      slope.append(temp[1]/temp[0])
    slope = np.array([slope]).reshape(-1,1)
    dev[c][i] = np.hstack((dev[c][i],slope))

# Add feature - Norm between consecutive points

for c in range(5):
  for i in range(len(train[c])):
    data = train[c][i]
    slope = []
    for j in range(len(data)):
      if j == len(data)-1 :
        temp = data[j] - data[j]
      else:
        temp = data[j+1]-data[j]
      slope.append(np.linalg.norm(temp))
    slope = np.array([slope]).reshape(-1,1)
    train[c][i] = np.hstack((train[c][i],slope))
  for i in range(len(dev[c])):
    data = dev[c][i]
    slope = []
    for j in range(len(data)):
      if j == len(data)-1 :
        temp = data[j] - data[j]
      else:
        temp = data[j+1]-data[j]
      slope.append(np.linalg.norm(temp))
    slope = np.array([slope]).reshape(-1,1)
    dev[c][i] = np.hstack((dev[c][i],slope))

C = 20 				# k-nearest neighbour

# Get the predicted classes
prediction = np.zeros((5,len(dev[0])))

for c in range(5):
	count = 0
	for d in dev[c]:
		cost = []
		for i in range(5):
			for t in train[i]:
				cost.append([DTW(d,t)[-1,-1], i])
		cost = np.array(cost)
		idx = cost[:,0].argsort()
		cost = cost[idx]
		prediction[c,count] = statistics.mode(cost.reshape(-1,2)[:C,1])
		count += 1

# Confusion Matrix

CMatrix = np.zeros((5,5))

for i in range(5):
	for j in range(5):
		CMatrix[j,i] = prediction[i][prediction[i]==j].shape[0]

# Plot heatmap for confusion matrix
sns.heatmap(CMatrix, annot = True, cmap ='plasma',linecolor ='black', linewidths = 1, fmt = '.0f', xticklabels = dataset, yticklabels = dataset)