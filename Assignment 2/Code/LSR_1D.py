import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

M = 10
N = 200

# Taking input from train and dev datasets
train_data = pd.read_csv("../Dataset/Regression/1D/1d_team_20_train.txt", sep = " ", header=None, names=["X","Y"])
dev_data = pd.read_csv("../Dataset/Regression/1D/1d_team_20_dev.txt", sep = " ", header=None, names=['XD','YD'])

idx = np.round(np.linspace(0, train_data.shape[0]-1, N).astype(int))

X = train_data.iloc[:,0].values[idx]
Y = train_data.iloc[:,1].values[idx]
XD = dev_data.iloc[:,0].values[idx]
YD = dev_data.iloc[:,1].values[idx]

A = np.ones((X.size, M+1))
for i in range(0,M+1):
	A[:, i] = X ** i

pseudo = np.linalg.pinv(A)
Alpha = (pseudo @ Y)[::-1]

# m degree polynomial with coefficients Alpha (Alpha[i] corresponds to the coefficient of x^(M - i))
model = np.poly1d(Alpha)

# # Plot for Training Dataset
# plt.scatter(X,Y, color = 'blue', alpha = 0.2, label = "Training Data")
# plt.plot(X, model(X),color = 'black', label = "Estimation")
# plt.legend()
# plt.show()

# Plot for Development Dataset
plt.title("Plot for M = " + str(M) + ", N = " + str(N))
plt.scatter(XD,YD, color = 'blue', alpha = 0.2, label = "Data")
plt.plot(XD, model(XD),color = 'black', label = "Estimation")
plt.legend()
plt.show()
