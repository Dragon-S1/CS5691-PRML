import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import griddata
import math
from Kmeans import *
from GMM import *

# fig = plt.figure(figsize=(10, 5))
# rows = 1
# cols = 2

#Reading the datasets
syth = np.loadtxt('../../Assets/K-Means/Synthetic Dataset/7/train.txt', delimiter=',')
# syth[:,:2] = (syth[:,:2] - np.mean(syth[:,:2], axis=0))/np.std(syth[:,:2], axis=0)
syth[:,:2] = (syth[:,:2] - syth[:,:2].min(axis = 0))/(syth[:,:2].max(axis = 0)-syth[:,:2].min(axis = 0))
syth_d = np.loadtxt('../../Assets/K-Means/Synthetic Dataset/7/dev.txt', delimiter=',')
syth_d[:,:2] = (syth_d[:,:2] - np.mean(syth_d[:,:2], axis=0))/np.std(syth_d[:,:2], axis=0)
C1 = syth[syth[:,2] == 1, :2]
C2 = syth[syth[:,2] == 2, :2]

def stats(x):
	μ = x.mean(0)
	Σ = np.cov(x, rowvar=False)
	return (μ, Σ)

def initial(x, K):
	n, d = x.shape
	μ_initial = np.zeros([K, d])
	Σ_initial = np.zeros([K, d, d])
	for k in range(K):
		index = random.randint(0, n)
		μ_initial[k,:] = x[index]
		Σ_initial[k,:,:] = stats(x)[1]
	return (μ_initial, Σ_initial)

## K Keans implementation
K = 16
μ1, Σ1 = initial(C1, K)
μ2, Σ2 = initial(C2, K)

N1, μ1, Σ1, dk1 = Kmeans(C1, μ1, K)
N2, μ2, Σ2, dk2 = Kmeans(C2, μ2, K)

for j in np.arange(1, 15):
	N1, μ1, Σ1, dk1 = Kmeans(C1, μ1, K)
	N2, μ2, Σ2, dk2 = Kmeans(C2, μ2, K)

# fig.add_subplot(rows,cols,1)
# plt.scatter(C1[:,0], C1[:,1], s=1, label ='Class 1')
# plt.scatter(C2[:,0], C2[:,1], s=1, label ='Class 2')
# plt.scatter(μ1[:,0], μ1[:,1], s=20, c = 'red', label ='Kmeans 1')
# plt.scatter(μ2[:,0], μ2[:,1], s=20, c = 'blue', label ='Kmeans 2')
# plt.title("Kmeans")
# plt.legend()

##GMM implementation
π1 = np.zeros(K)
π2 = np.zeros(K)
for k in range(K):
	π1[k] = N1[k]/C1.shape[0]
	π2[k] = N2[k]/C2.shape[0]

π1, μ1, Σ1, dg1 = GMM(C1, π1, μ1, Σ1, K)
π2, μ2, Σ2, dg2 = GMM(C2, π2, μ2, Σ2, K)

for j in range(1, 2):
	π1, μ1, Σ1, dg1 = GMM(C1, π1, μ1, Σ1, K)
	π2, μ2, Σ2, dg2 = GMM(C2, π2, μ2, Σ2, K)

# print(dk1, dk2)
# print(dg1, dg2)

# fig.add_subplot(rows,cols,2)
# plt.scatter(C1[:,0], C1[:,1], s=1, label ='Class 1')
# plt.scatter(C2[:,0], C2[:,1], s=1, label ='Class 2')
# plt.scatter(μ1[:,0], μ1[:,1], s=20, c = 'red', label ='GMM 1')
# plt.scatter(μ2[:,0], μ2[:,1], s=20, c = 'blue', label ='GMM 2')
# plt.title("GMM")
# plt.legend()

# plt.show()

#grid function creates sets of points (x,y) over a given region
def grid(x, n):
	min1, max1 = x[:, 0].min(), x[:, 0].max()
	min2, max2 = x[:, 1].min(), x[:, 1].max()
	x1grid = np.linspace(min1, max1, n)
	x2grid = np.linspace(min2, max2, n)
	xx, yy = np.meshgrid(x1grid, x2grid)
	xx, yy = np.meshgrid(x1grid, x2grid)
	r1, r2 = xx.flatten(), yy.flatten()
	r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
	grid = np.hstack((r1,r2))
	return (xx,yy,grid)
 
#Decision boundary function as plotted using z value of the grid
def decision(x, ytest, n):
	xx, yy = grid(x, n)[:2]
	z = ytest.reshape(xx.shape)
	return (xx, yy, z)

N = 80 #no. of points used to make each grid dimension before meshgrid

#1)Contour lines
xx1, yy1, g1 = grid(C1[:,:2], N)
xx2, yy2, g2 = grid(C2[:,:2], N)
GDF1 = np.zeros((K,len(g1)))
GDF2 = np.zeros((K,len(g2)))
for k in range(K):
  for i in range(len(g1)):
    GDF1[k, i] = gaussian(g1[i], μ1[k], Σ1[k])
  for i in range(len(g2)):
    GDF2[k, i] = gaussian(g2[i], μ2[k], Σ2[k])

#removing unwanted GDF=0 values as they produce incorrect contours
GDF1[np.where(GDF1 == 0)] = math.exp(-200)
GDF2[np.where(GDF2 == 0)] = math.exp(-200)

for k in range(K):
	zz1 = GDF1[k,:].reshape(xx1.shape)
	zz2 = GDF2[k,:].reshape(xx2.shape)
	plt.contour(xx1, yy1, zz1, linewidths=0.4, cmap='Paired')
	plt.contour(xx2, yy2, zz2, linewidths=0.4, cmap='Paired')   

#2)Decision Boundary
xtest = grid(syth[:, :2], N)[2]   
P1 = np.zeros(len(xtest)); P2 = np.zeros(len(xtest))
for i in range(len(xtest)):
  for k in range(K):
    P1[i] += π1[k]*gaussian(xtest[i], μ1[k], Σ1[k])
    P2[i] += π2[k]*gaussian(xtest[i], μ2[k], Σ2[k])
P = np.vstack((P1, P2)).T

#making preditions on xtest
ytest = []; yscore =[]
for i in range(len(xtest)):
  ytest.append(np.argmax(P[i,:])+1)
  yscore.append(np.max(P[i,:]))
ytest = np.array(ytest); yscore = np.array(yscore)

xx, yy, z = decision(syth[:, :2], ytest, N)
plt.contourf(xx, yy, z, colors = ['springgreen','lightskyblue'], alpha = .5)

#3)GMM plots
plt.scatter(C1[:,0], C1[:,1], s=1, label ='Class 1')
plt.scatter(C2[:,0], C2[:,1], s=1, label ='Class 2')
plt.scatter(μ1[:,0], μ1[:,1], s=20, c = 'red')
plt.scatter(μ2[:,0], μ2[:,1], s=20, c = 'blue')
plt.title('GMM')
plt.legend()

plt.show()