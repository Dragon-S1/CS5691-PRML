import numpy as np

#KMeans function parameters estimation
def Kmeans(x, μ, K):
	n_tot, d = x.shape
	err = []
	N = np.zeros(K)
	distribution = np.zeros([n_tot, K])
	#finding the closest centroid to each data point
	for n in range(n_tot):
		for k in range(K):
			err.append(np.linalg.norm(μ[k]-x[n]))
		err = np.array(err)
		i = np.argmin(err)
		N[i] += 1
		distribution[n, i] = 1
		err = []
	# estimating the new mean
	μ_new = np.zeros([K, d])
	for k in range(K):
		for n in range(n_tot):
			μ_new[k, :] += x[n]*distribution[n, k]
		μ_new[k, :] /= N[k]

	Σ_new = np.zeros([K, d, d])
	distortion = 0
	# estimating the new covariance
	for k in range(K):
		for n in range(n_tot):
			Σ_new[k, :, :] += distribution[n, k]*((x[n]-μ_new[k]).reshape(d,1)@(x[n]-μ_new[k]).reshape(1,d))
			distortion += distribution[n, k]*np.linalg.norm(μ_new[k]-x[n])
		Σ_new[k, :, :] /= N[k]

	return (N, μ_new, Σ_new, distortion)