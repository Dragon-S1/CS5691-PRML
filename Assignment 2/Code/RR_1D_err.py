import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

M = 10

# Taking input from train and dev datasets
train_data = pd.read_csv("../Dataset/Regression/1D/1d_team_20_train.txt", sep = " ", header=None, names=["X","Y"])
dev_data = pd.read_csv("../Dataset/Regression/1D/1d_team_20_dev.txt", sep = " ", header=None, names=['XD','YD'])

X = train_data.iloc[:,0].values
Y = train_data.iloc[:,1].values
XD = dev_data.iloc[:,0].values
YD = dev_data.iloc[:,1].values

# store the root mean square error in Y, YD
Y_RMSE = []
YD_RMSE = []

A = np.ones((X.size, M+1))
for i in range(0,M+1):
	A[:, i] = X ** i

AD = np.ones((XD.size, M+1))
for i in range(0,M+1):
	AD[:, i] = XD ** i


# X_L -> range of ln(Lamda) for error graph
X_L = range(-30, 1)

# calculating rmse corresponding to different values of ln(Lamda)
for i in X_L:
	Lambda = (math.exp(i))
	D = np.ones(M+1) * Lambda

	pseudo = np.linalg.inv((A.T @ A) + np.diag(D)) @ A.T
	Alpha = (pseudo @ Y)[::-1]

	# m degree polynomial with coefficients Alpha (Alpha[i] corresponds to the coefficient of x^(M - i))
	model = np.poly1d(Alpha)

	Y_RMSE.append(np.linalg.norm(Y - model(X)) / len(Y))
	YD_RMSE.append(np.linalg.norm(YD - model(XD)) / len(YD))

# plotting the Error graph
plt.title("Plot for M = " + str(M) + ", N = 200 and varying ln(lambda)")
plt.xlabel("ln of regularization parameter")
plt.ylabel("Root Mean Square Error")
plt.plot(X_L, Y_RMSE, color = 'blue', label = "Training", marker = 'o', markersize = 4, markeredgecolor = 'blue', markerfacecolor = 'white')
plt.plot(X_L, YD_RMSE, color = 'red', label = "Development", marker = 'o', markersize = 4, markeredgecolor = 'red', markerfacecolor = 'white')
plt.plot(X_L[np.argmin(Y_RMSE)], Y_RMSE[np.argmin(Y_RMSE)], marker="o", markersize=4, color= 'blue', label = 'Min of Training (ln(lambda) = ' + str(X_L[np.argmin(Y_RMSE)]) + ')')
plt.plot(X_L[np.argmin(YD_RMSE)], YD_RMSE[np.argmin(YD_RMSE)], marker="o", markersize=4, color = 'red', label = 'Min of Development (ln(lambda) = ' + str(X_L[np.argmin(YD_RMSE)]) + ')')
plt.legend()
plt.show()
