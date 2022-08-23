import numpy as np
from DTW_Digit_Read import *
from DTW import *
import statistics
import seaborn as sns

train = []
dev = []

# Given Datasets
dataset = ['2', '3', '4', '5', 'z']

# Read Data
for i in dataset:
	train.append(read_data(i, 'train', '.mfcc'))
	dev.append(read_data(i, 'dev', '.mfcc'))


C = 25 				# K-nearest neighbours

# Get the predicted classes

prediction = np.zeros((5,len(dev[0])))

for c in range(5):
	count = 0
	for d in dev[c]:
		cost = []
		for i in range(5):
			for t in train[i]:
				cost.append([DTW(d,t)[-1,-1], i]) #Apply DWT on test data from each train data
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

# Plot heatmap of confusion matrix
sns.heatmap(CMatrix, annot = True, cmap ='plasma',linecolor ='black', linewidths = 1, fmt = '.0f', xticklabels='2345z', yticklabels='2345z')