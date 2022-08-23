import numpy as np

#standarad multivaraite gaussian density function
def gaussian(x, μ, Σ):
	d = len(μ)
	exponent = np.exp(-0.5*(x-μ)@np.linalg.inv(Σ)@(x-μ).T)
	det = 1/(np.linalg.det(Σ))**0.5
	gdf = 1/(2*np.pi)**(d/2)*det*exponent
	return gdf

#Gaussian Mixture Model parameter estimation function
def GMM(x, π, μ, Σ, K):
	n_tot, d = x.shape
	γ = np.zeros((n_tot, K))
	# estimating γ responsibility
	for n in range(n_tot):                            
		Sum = 0
		for k in range(K):
			Sum += π[k]*gaussian(x[n], μ[k], Σ[k])
		for k in range(K):
			γ[n, k] = (π[k]*gaussian(x[n], μ[k], Σ[k]))/Sum 

	N = np.zeros(K)
	π_new = np.zeros(K)
	# estimating the weights π 
	for k in range(K):
		for n in range(n_tot):
			N[k] += γ[n, k]
		π_new[k] = N[k]/n_tot
	μ_new = np.zeros((K, d))
	# estimating the new mean
	for k in range(K):
		for n in range(n_tot):
			μ_new[k, :] += γ[n, k]*x[n]
		μ_new[k, :] /= N[k]

	Σ_new = np.zeros([K, d, d])
	distortion = 0
	# estimating the new covariance
	for k in range(K):
		for n in range(n_tot):
			Σ_new[k, :, :] += γ[n, k] * (x[n]-μ_new[k]).reshape(d,1)@(x[n]-μ_new[k]).reshape(1,d)
			distortion += γ[n, k]*np.linalg.norm(μ_new[k]-x[n])
		Σ_new[k, :, :] /= N[k]
    # Uncomment the below lines for a diaginal covariance estimation
#     for K in np.arange(K):
#         Σ_new[K, :, :] = np.eye(d)*Σ_new[K, :, :]

	return (π_new, μ_new, Σ_new, distortion)