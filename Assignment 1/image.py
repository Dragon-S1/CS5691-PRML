import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# CONFIG

np.set_printoptions(precision=4, suppress=True)

# setup for image plot
fig = plt.figure(figsize=(10, 7))
rows = 1
cols = 4

# input the value of k
k = 200


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

# only take top k values
np.fill_diagonal(d[k:,k:], 0)

# reconstruct the image
r_img_evd = p @ d @ p_inv
r_img_evd = np.abs(r_img_evd)

# plot the reconstructed image and error image
fig.add_subplot(rows,cols,1)
plt.imshow(r_img_evd, cmap='gray')
plt.axis('off')
plt.title("Image of top " +str(k)+ " eigenvalues", fontsize = 8)
fig.add_subplot(rows,cols,2)
plt.imshow(img - r_img_evd, cmap='gray')
plt.axis('off')
plt.title("Error image of top " +str(k)+ " eigenvalues", fontsize = 8)


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

# get the sigma matrix
sigma = u.T @ img @ v
# take only top k values
np.fill_diagonal(sigma[k:,k:], 0)

# reconstruct the image
r_img_svd = u @ sigma @ v.T

# plot the recosntructed image and error image
fig.add_subplot(rows,cols,3)
plt.imshow(r_img_svd, cmap='gray')
plt.axis('off')
plt.title("Image of top " +str(k)+ " singular values", fontsize = 8)
fig.add_subplot(rows,cols,4)
plt.imshow((img-r_img_svd), cmap='gray')
plt.axis('off')
plt.title("Error image of top " +str(k)+ " singular values", fontsize = 8)
plt.show()