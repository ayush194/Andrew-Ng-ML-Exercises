import math
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

data = loadmat('data/bird_small.mat')
#normalize values
tmp = data['A'] / 255
m, n, p = tmp.shape[0], tmp.shape[1], tmp.shape[2]
x = tmp.reshape((m * n, p))

def assignCentroids(x, μ):
	m = x.shape[0]
	c = np.zeros((m,))
	k = μ.shape[0]
	for i in range(m):
		min_dist = 1000000
		for j in range(k):
			dist = np.linalg.norm(x[i] - μ[j])
			if (dist < min_dist):
				min_dist = dist
				c[i] = j
	return c

def moveCentroids(x, c, k):
	m = x.shape[0]
	n = x.shape[1]
	μ = np.zeros((k, n))
	for i in range(k):
		tmp = np.array([x[ci] for ci in range(m) if c[ci] == i])
		if (tmp.shape[0] == 0):
			continue
		μ[i] = (tmp.sum(axis=0)) / np.float64(tmp.shape[0])
	return μ

def randomInit(x, k):
	#μ = np.array([x[int(np.random.rand() * k)] for i in range(k)])
	#the following is a better way of selecting random datasets from x
	#because using the above statement may select the same data point twice
	tmp = np.copy(x)
	np.random.shuffle(tmp)
	μ = tmp[:k,:]
	return μ

def formClusters(x, k, max_iters):
	#randomly initialze μ
	μ = randomInit(x, k)
	for i in range(max_iters):
		c = assignCentroids(x, μ)
		μ = moveCentroids(x, c, k)
	return μ, c

μ, c = formClusters(x, 16, 10)
for i in range(x.shape[0]):
	x[i] = μ[int(c[i])]
recovered_img = x.reshape(m, n, p)

plt.imshow(recovered_img)
plt.show()