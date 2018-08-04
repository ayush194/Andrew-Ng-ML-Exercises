import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.io import loadmat

data = loadmat('data/ex8data1.mat')
x = data['X']
x_val = data['Xval']
y_val = data['yval'][:,0]
print(x.shape)

def findGaussian(x):
	return x.mean(axis=0), x.var(axis=0)

def probabilities(x, mean, var):
	#x is an m*n dimensional matrix
	m = x.shape[0]
	n = x.shape[1]
	p = np.ones(x.shape)
	for i in range(n):
		tmp = stats.norm(mean[i], var[i])
		p[:,i] = tmp.pdf(x[:,i])
	prob = np.prod(p, axis=1)
	return prob

def fScore(y_pred, y_val):
	#tp, tn, fn, fp are true positive, true negative, false negative, false positive
	tp = np.sum(np.logical_and(y_pred == 1, y_val == 1)).astype(float)
	tn = np.sum(np.logical_and(y_pred == 0, y_val == 0)).astype(float)
	fn = np.sum(np.logical_and(y_pred == 0, y_val == 1)).astype(float)
	fp = np.sum(np.logical_and(y_pred == 1, y_val == 0)).astype(float)
	#precision and recall
	if (tp + fp == 0 or tp + fn == 0):
		return -1
	p = tp / (tp + fp)
	r = tp / (tp + fn)
	f_score = (2 * p * r) / (p + r)
	#print(f_score)
	return f_score

def findThreshold(x_val, y_val, mean, var):
	prob = probabilities(x_val, mean, var)
	epsilon_best, f_best = 0, 0
	step = (prob.max() - prob.min()) / 1000
	for i in np.arange(prob.min(), prob.max(), step):
		epsilon = i
		y_pred = np.array([1 if el < epsilon else 0 for el in prob])
		f = fScore(y_pred, y_val)
		if (f > f_best):
			epsilon_best, f_best = epsilon, f
	return epsilon_best, f_best

mean, var = findGaussian(x)
epsilon, f = findThreshold(x_val, y_val, mean, var)
prob = probabilities(x, mean, var)
print(epsilon, f)
#print(epsilon, f)
outliers = np.array([x[i] for i in range(x.shape[0]) if (prob[i] < epsilon)])
print(outliers.shape)

#fig, ax = plt.subplots(figsize=(12,8))
#ax.scatter(x[:,0], x[:,1])
#ax.scatter(outliers[:,0], outliers[:,1])
#plt.show()