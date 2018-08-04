import math
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

data = loadmat('data/ex7data1.mat')
x = data['X']
m = x.shape[0]

def pca(x):
	m = x.shape[0]
	#feature scaling
	x_scaled = (x - x.mean()) / x.std()
	covarience = covarience = (x_scaled.T @ x_scaled) / m
	u, s, v = np.linalg.svd(covarience)
	return u, s, v, x_scaled, x.mean(), x.std()

def project(x, u, k):
	u_reduced = u[:,:k]
	z = x @ u_reduced
	return z

def recover(z, u, k):
	u_reduced = u[:,:k]
	x_recovered = z @ u_reduced.T
	return x_recovered

u, s, v, x_scaled, x_mean, x_std = pca(x)
z = project(x, u, 1)
x_recovered = recover(z, u, 1)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x_recovered[:, 0], x_recovered[:, 1])
plt.show()