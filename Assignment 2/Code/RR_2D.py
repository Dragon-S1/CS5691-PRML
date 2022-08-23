from functools import partial
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
N = 1000
lnLambda = 0

Lambda = math.exp(lnLambda)

# Taking input from train and dev datasets
train_data = pd.read_csv("../Dataset/Regression/2D/2d_team_20_train.txt", sep = " ", header=None, names=["X1","X2", "Y"])
dev_data = pd.read_csv("../Dataset/Regression/2D/2d_team_20_dev.txt", sep = " ", header=None, names=['X1D',"X2D" ,'YD'])

idx = np.round(np.linspace(0, train_data.shape[0]-1, N).astype(int))

X1 = train_data.iloc[:,0].values[idx]
X2 = train_data.iloc[:,1].values[idx]
Y = train_data.iloc[:,2].values[idx]
X1D = dev_data.iloc[:,0].values[idx]
X2D = dev_data.iloc[:,1].values[idx]
YD = dev_data.iloc[:,2].values[idx]

# M_ -> number of terms in polynomial
M_ = int(((M+1)*(M+2)) / 2)
A = np.ones((X1.size, M_))
count = 0
for i in range(0,M+1):
    for j in range(0, i+1):
        A[:, count] = (X1 ** (i - j)) * (X2 ** j)
        count += 1

D = np.ones(M_) * Lambda
pseudo = np.linalg.inv((A.T @ A) + np.diag(D)) @ A.T
Alpha = (pseudo @ Y)

model = partial(Y_func, Alpha = Alpha, M = M)

# Plot for Training data
# ax = plt.axes(projection ='3d')
# ax.scatter(X1, X2, Y)
# ax.plot_trisurf(X1, X2, model(X1, X2), color = "blue")

# Plot for Dev Data
ax = plt.axes(projection ='3d')
plt.title("Plot for N = "+str(N)+" and M ="+str(M)+" and ln(lambda) =" + str(lnLambda))
ax.scatter(X1D, X2D, YD, color = 'red', alpha = 0.2)
ax.plot_trisurf(X1D, X2D, model(X1D, X2D), color = "blue", alpha = 0.8)

plt.show()