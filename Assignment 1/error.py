import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# CONFIG

np.set_printoptions(precision=4, suppress=True)
# initailise index array for each k
index = np.arange(0,256)


### EVD ###

# read the image
img = mpimg.imread('64.jpg')

# eigenvalue decomposition
eigvalues, eigvectors = np.linalg.eig(img)

# sort the eigenvalues and eigenvectors
idx = np.abs(eigvalues).argsort()[::-1]
p = eigvectors[:,idx]

# get the inverse of matrix p
p_inv = np.linalg.inv(p)
# get the diagonal matrix
d = np.diag(eigvalues[idx])

err_evd = np.zeros(256)

for k in range(0,256):
	# get the diagonal matrix
	d = np.diag(eigvalues[idx])
	# only take top k values
	np.fill_diagonal(d[k:,k:], 0)
	# reconstruct the image
	r_img_evd = p @ d @ p_inv
	r_img_evd = np.abs(r_img_evd)
	# get the reconstruction error for k
	err_evd[k] = (np.linalg.norm(img-r_img_evd))/np.linalg.norm(img)

# plot the Reconstruction error vs. K graph
plt.plot(index[1:], err_evd[1:], label = "EVD")
plt.title("Reconstruction error vs. K")
plt.xlabel("k")
plt.ylabel("Reconstruction error")


### SVD ###

# scale the image for svd
img = (img-img.mean())/img.std()

# do eigenvalue decomposition of dot products of img and its transpose
u_eig_val, u_eig_vec = np.linalg.eig(img @ img.T)
v_eig_val, v_eig_vec = np.linalg.eig(img.T @ img)

# sort the values
u_idx = u_eig_val.argsort()[::-1]
v_idx = v_eig_val.argsort()[::-1]

# sort the vectors and store in u and v
u = u_eig_vec[:,u_idx]
v = v_eig_vec[:,v_idx]

err_svd = np.zeros(256)

for k in range(0,256):
	# get the sigma matrix
	sigma = u.T @ img @ v
	# take only top k values
	np.fill_diagonal(sigma[k:,k:], 0)
	# reconstruct the image
	r_img_svd = u @ sigma @ v.T
	# get the reconstruction error for k
	err_svd[k] = (np.linalg.norm(img-r_img_svd))/np.linalg.norm(img)

# plot the Reconstruction error vs. K graph
plt.plot(index[1:], err_svd[1:], label = "SVD")
plt.legend()
plt.show()