import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import det_curve
import seaborn as sn

# np.set_printoptions(precision=6, suppress=True)


# Function to calculate probability density given data, mean and covariance matrix
def p_i(x, mu, sigma):
    sqrtOf_det_sigma = np.linalg.det(sigma)**(1/2)
    inv_sigma = np.linalg.inv(sigma)
    result = (2 * math.pi * sqrtOf_det_sigma)**(-1) * math.exp((-1/2) * ((x - mu).T @ inv_sigma @ (x - mu)))
    return result

# import train and dev data
train_data = pd.read_csv("../Dataset/Bayesian Classifier/Data/RealData/train.txt", sep = ",", header=None, names=["X","Y","C"])
dev_data = pd.read_csv("../Dataset/Bayesian Classifier/Data/RealData/dev.txt", sep = ",", header=None, names=['XD','YD', 'CD'])

# Number for data points
N = 1050

# train
idx_train = np.round(np.linspace(0, (train_data.shape[0]) - 1, N)).astype(int)

# dev
idx_dev = np.round(np.linspace(0, dev_data.shape[0] - 1, N)).astype(int)

# Separate the classes from train data
tData = np.array(train_data.iloc[:].values[idx_train])
w1 = tData[tData[:,2] == 1]
w2 = tData[tData[:,2] == 2]
w3 = tData[tData[:,2] == 3]

# Separate the classes from dev data
dData = dev_data.iloc[:].values[idx_dev]
w1D = dData[dData[:,2] == 1]
w2D = dData[dData[:,2] == 2]
w3D = dData[dData[:,2] == 3]

# Calculate the mean of each class
mu1 = np.mean(w1[:,0:2], axis = 0)
mu2 = np.mean(w2[:,0:2], axis = 0)
mu3 = np.mean(w3[:,0:2], axis = 0)

# case 2
C1 = np.cov(np.stack(w1[:,0:2], axis=1))
C2 = np.cov(np.stack(w2[:,0:2], axis=1))
C3 = np.cov(np.stack(w3[:,0:2], axis=1))

# case 1
C = (C1 + C2 + C3)/3

# case 3
sigma2 = np.var(tData[:,0:2])
CS1 = sigma2 * np.identity(2)
CS2 = sigma2 * np.identity(2)
CS3 = sigma2 * np.identity(2)

# Get the predicted data
w = np.array([1+np.argmax(np.array([p_i(x,mu1,C), p_i(x,mu2,C), p_i(x,mu3,C)])) for x in dData[:,0:2]])

# probability density function of each classes
pdf1 = np.array([p_i(x,mu1,C1) for x in w1D[:,0:2]])
pdf2 = np.array([p_i(x,mu2,C2) for x in w2D[:,0:2]])
pdf3 = np.array([p_i(x,mu3,C3) for x in w3D[:,0:2]])

# PDF plot
ax = plt.axes(projection ='3d')
surf1 = ax.plot_trisurf(w1D[:,0], w1D[:,1], pdf1, color = "red", alpha = 0.8, label = 'Class 1')
surf2 = ax.plot_trisurf(w2D[:,0], w2D[:,1], pdf2, color = "blue", alpha = 0.8, label = 'Class 2')
surf3 = ax.plot_trisurf(w3D[:,0], w3D[:,1], pdf3, color = "green", alpha = 0.8, label = 'Class 3')
surf1._edgecolors2d = surf1._edgecolor3d
surf1._facecolors2d = surf1._facecolor3d
surf2._edgecolors2d = surf2._edgecolor3d
surf2._facecolors2d = surf2._facecolor3d
surf3._edgecolors2d = surf3._edgecolor3d
surf3._facecolors2d = surf3._facecolor3d
plt.legend()
plt.show()


# DET Curve Plot

T = np.array(w == dData[:,2]).astype(int)
S = np.array([pdf1,pdf2,pdf3])
S = S.reshape(1050,)
fpr, fnr, thresholds = det_curve(T, S)
plt.plot(fpr*100,fnr*100, label = "Case 2")

pdf1 = np.array([p_i(x,mu1,C) for x in w1D[:,0:2]])
pdf2 = np.array([p_i(x,mu2,C) for x in w2D[:,0:2]])
pdf3 = np.array([p_i(x,mu3,C) for x in w3D[:,0:2]])

T = np.array(w == dData[:,2]).astype(int)
S = np.array([pdf1,pdf2,pdf3])
S = S.reshape(1050,)
fpr, fnr, thresholds = det_curve(T, S)
plt.plot(fpr*100,fnr*100, label = "Case 1")

pdf1 = np.array([p_i(x,mu1,CS1) for x in w1D[:,0:2]])
pdf2 = np.array([p_i(x,mu2,CS2) for x in w2D[:,0:2]])
pdf3 = np.array([p_i(x,mu3,CS3) for x in w3D[:,0:2]])

T = np.array(w == dData[:,2]).astype(int)
S = np.array([pdf1,pdf2,pdf3])
S = S.reshape(1050,)
fpr, fnr, thresholds = det_curve(T, S)

plt.plot(fpr*100,fnr*100, label = "Case 3")
plt.xlabel("False Alarm rate in %")
plt.ylabel("Missed Detection rate in %")
plt.legend()
plt.show()

# Confusion matrix

CM = np.zeros((3,3))
for i in range(0,int(N/3)):
		CM[0,w[i]-1] += 1
for i in range(int(N/3),int(2*N/3)):
		CM[1,w[i]-1] += 1
for i in range(int(2*N/3),N):
		CM[2,w[i]-1] += 1

df_cm = pd.DataFrame(CM, index = [i for i in "123"],
                  columns = [i for i in "123"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()