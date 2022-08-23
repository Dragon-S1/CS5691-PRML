import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# calculate the value of Y for X1, X2 by substituting values of X1, X2 in polynomial function for 2d
def Y_func(X1, X2, Alpha, M):
    Y = np.zeros(X1.size)
    count = 0
    for i in range(0,M):
        for j in range(0, i+1):
            Y += Alpha[count] * (X1 ** (i - j)) * (X2 ** j)
            count += 1
    return Y

M = 7

# Taking input from train and dev datasets
train_data = pd.read_csv("../Dataset/Regression/2D/2d_team_20_train.txt", sep = " ", header=None, names=["X1","X2", "Y"])
dev_data = pd.read_csv("../Dataset/Regression/2D/2d_team_20_dev.txt", sep = " ", header=None, names=['X1D',"X2D" ,'YD'])

X1 = train_data.iloc[:,0].values
X2 = train_data.iloc[:,1].values
Y = train_data.iloc[:,2].values
XD1 = dev_data.iloc[:,0].values
XD2 = dev_data.iloc[:,1].values
YD = dev_data.iloc[:,2].values

# store the root mean square error in Y, YD, YT 
Y_RMSE = []
YD_RMSE = []

M_ = int(((M+1)*(M+2)) / 2)
A = np.ones((X1.size, M_))
count = 0
for i in range(0,M+1):
    for j in range(0, i+1):
        A[:, count] = (X1 ** (i - j)) * (X2 ** j)
        count += 1

# X_L -> range of ln(Lamda) for error graph
X_L = range(-30, 5)

# calculating rmse corresponding to different values of ln(Lamda)
for i in X_L:
	Lambda = (math.exp(i))
	D = np.ones(M_) * Lambda

	pseudo = np.linalg.inv((A.T @ A) + np.diag(D)) @ A.T
	Alpha = (pseudo @ Y)

	Y_RMSE.append(np.linalg.norm(Y - Y_func(X1, X2, Alpha, M)) / len(Y))
	YD_RMSE.append(np.linalg.norm(YD - Y_func(XD1, XD2, Alpha, M)) / len(YD))

# plotting the Error graph
plt.title("Error plot for M=" + str(M) + " N=1000 and varying ln(lambda)")
plt.plot(X_L, Y_RMSE, color = 'blue', label = "Training", marker = 'o', markersize = 4, markeredgecolor = 'blue', markerfacecolor = 'white')
plt.plot(X_L, YD_RMSE, color = 'red', label = "Development", marker = 'o', markersize = 4, markeredgecolor = 'red', markerfacecolor = 'white')
plt.plot(X_L[np.argmin(Y_RMSE)], Y_RMSE[np.argmin(Y_RMSE)], marker="o", markersize=4, color= 'blue', label = 'Min of Training at ln(lambda) = ' + str(X_L[np.argmin(Y_RMSE)]))
plt.plot(X_L[np.argmin(YD_RMSE)], YD_RMSE[np.argmin(YD_RMSE)], marker="o", markersize=4, color = 'red', label = 'Min of Development at ln(lambda) = ' + str(X_L[np.argmin(YD_RMSE)]))
plt.legend()
plt.show()