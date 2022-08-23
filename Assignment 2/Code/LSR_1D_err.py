import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Taking input from train and dev datasets
train_data = pd.read_csv("../Dataset/Regression/1D/1d_team_20_train.txt", sep = " ", header=None, names=["X","Y"])
dev_data = pd.read_csv("../Dataset/Regression/1D/1d_team_20_dev.txt", sep = " ", header=None, names=['XD','YD'])

X = train_data.iloc[:,0].values
Y = train_data.iloc[:,1].values
XD = dev_data.iloc[:,0].values
YD = dev_data.iloc[:,1].values

# store the root mean square error in Y, YD, YT
Y_RMSE = []
YD_RMSE = []

# X_M -> range of M on error graph
X_M = range(1, 31)


# calculating rmse corresponding to different values of M
for M in X_M:
	A = np.ones((X.size, M+1))
	for i in range(0,M+1):
		A[:, i] = X ** i

	pseudo = np.linalg.pinv(A)
	Alpha = (pseudo @ Y)[::-1]

	# m degree polynomial with coefficients Alpha (Alpha[i] corresponds to the coefficient of x^(M - i))
	model = np.poly1d(Alpha)

	Y_RMSE.append(np.linalg.norm(Y - model(X)) / len(Y))
	YD_RMSE.append(np.linalg.norm(YD - model(XD)) / len(YD))

# plotting the Error graph
plt.title("Error plot for N = 200 and varying M")
plt.xlabel("Model Complexity")
plt.ylabel("Root Mean Square Error")
plt.plot(X_M, Y_RMSE, color = 'blue', label = "Training", marker = 'o', markersize = 4, markeredgecolor = 'blue', markerfacecolor = 'white')
plt.plot(X_M, YD_RMSE, color = 'red', label = "Development", marker = 'o', markersize = 4, markeredgecolor = 'red', markerfacecolor = 'white')
plt.plot(X_M[np.argmin(Y_RMSE)], Y_RMSE[np.argmin(Y_RMSE)], marker="o", markersize=4, color= 'blue', label = 'Min of Training (M = ' + str(X_M[np.argmin(Y_RMSE)]) + ')')
plt.plot(X_M[np.argmin(YD_RMSE)], YD_RMSE[np.argmin(YD_RMSE)], marker="o", markersize=4, color = 'red', label = 'Min of Development (M = ' + str(X_M[np.argmin(YD_RMSE)]) + ')')
plt.legend()
plt.show()
