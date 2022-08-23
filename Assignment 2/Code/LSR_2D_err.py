import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# calculate the value of Y for X1, X2 by substituting values of X1, X2 in polynomial function for 2d
def Y_func(X1, X2, Alpha, M):
    Y = np.zeros(X1.size)
    count = 0
    for i in range(0,M):
        for j in range(0, i+1):
            Y += Alpha[count] * (X1 ** (i - j)) * (X2 ** j)
            count += 1
    return Y

# Taking input from train and dev datasets
train_data = pd.read_csv("../Dataset/Regression/2D/2d_team_20_train.txt", sep = " ", header=None, names=["X1","X2", "Y"])
dev_data = pd.read_csv("../Dataset/Regression/2D/2d_team_20_dev.txt", sep = " ", header=None, names=['X1D',"X2D" ,'YD'])

X1 = train_data.iloc[:,0].values
X2 = train_data.iloc[:,1].values
Y = train_data.iloc[:,2].values
X1D = dev_data.iloc[:,0].values
X2D = dev_data.iloc[:,1].values
YD = dev_data.iloc[:,2].values

# store the root mean square error in Y, YD, YT 
Y_RMSE = []
YD_RMSE = []

# X_M -> range of M on error graph
X_M = range(1, 15)

# calculating rmse corresponding to different values of M
for M in X_M:
    M_ = int(((M+1)*(M+2)) / 2)
    A = np.ones((X1.size, M_))
    count = 0
    for i in range(0,M+1):
        for j in range(0, i+1):
            A[:, count] = (X1 ** (i - j)) * (X2 ** j)
            count += 1

    pseudo = np.linalg.pinv(A)
    Alpha = (pseudo @ Y)

    Y_RMSE.append(np.linalg.norm(Y - Y_func(X1, X2, Alpha, M)) / len(Y))
    YD_RMSE.append(np.linalg.norm(YD - Y_func(X1D, X2D, Alpha, M)) / len(YD))


# plotting the Error graph
plt.title("Error plots for N = 1000 data points and varying M")
plt.plot(X_M, Y_RMSE, color = 'blue', label = "Training", marker = 'o', markersize = 4, markeredgecolor = 'blue', markerfacecolor = 'white')
plt.plot(X_M, YD_RMSE, color = 'red', label = "Development", marker = 'o', markersize = 4, markeredgecolor = 'red', markerfacecolor = 'white')
plt.plot(X_M[np.argmin(Y_RMSE)], Y_RMSE[np.argmin(Y_RMSE)], marker="o", markersize=4, color= 'blue', label = 'Min of Training (M = ' + str(X_M[np.argmin(Y_RMSE)]) + ')')
plt.plot(X_M[np.argmin(YD_RMSE)], YD_RMSE[np.argmin(YD_RMSE)], marker="o", markersize=4, color = 'red', label = 'Min of Development (M = ' + str(X_M[np.argmin(YD_RMSE)]) + ')')
plt.legend()
plt.show()