import math
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

data = loadmat('data/ex3data1.mat')
theta = loadmat('data/ex3weights.mat')
theta1 = theta['Theta1']
theta2 = theta['Theta2']

'''
tmp = np.ones(((data['X'].shape)[0],1), dtype=np.float64)
x = np.insert(data['X'], [0], tmp, axis=1)
y_prime = data['y'].transpose()[0]
y = np.array([elem if elem != 10 else 0 for elem in y_prime])

a1 = theta1 @ x.transpose()
print(a1)
a1 = np.insert(a1, [0], np.ones((a1.shape[1],), dtype=np.float64), axis=0)
a2 = (theta2 @ a1).transpose()
print(a2[2000])
print(y[2000])
'''

theta1_tmp = theta1.flatten(order='C')
theta2_tmp = theta2.flatten(order='C')
theta_tmp = np.concatenate((theta1_tmp, theta2_tmp), axis=0)
#print(theta1)
#print(theta2)
#print(theta1_tmp)
#print(theta2_tmp)
#print(theta_tmp.shape)
layers_tmp = np.array([400, 25, 10])
#print(layers_tmp.shape)

def roll(theta, layers):
	#returns a list of rolled theta matrices
	rolled_theta = []
	offset = 0
	for i in range(1, layers.shape[0]):
		m, n = layers[i], layers[i - 1] + 1
		rolled_theta.append(theta[offset: offset + m * n].reshape((m, n)))
		offset += m * n
	return rolled_theta

def feedForward(theta, layers):
	return

def unroll(thetas):
	unrolled_theta = np.concatenate(tuple([theta.flatten() for theta in thetas]), axis=0)
	return unrolled_theta
	#print(unrolled_theta)
	#flattens out(unrolls) a tuple of thetas
	#unrolled_theta = np.concatenate(theta)

unroll((theta1, theta2))
theta = roll(unroll((theta1, theta2)), layers_tmp)
print(theta[0])
print(theta[1])


