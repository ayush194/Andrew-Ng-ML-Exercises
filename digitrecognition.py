import math
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

data = loadmat('data/ex3data1.mat')

def g(z):
	#sigmoid function
	return 1.0 / (1 + math.pow(math.e, -z))
	
g = np.vectorize(g, otypes=[np.float64])
#now, when g is applied to a vector, it is applied to each cell

def J_Reg(theta, x, y, lamda):
	#using the vector notation
	ones = np.ones((y.shape[0],), dtype=np.float64)
	h = g(x @ theta)
	tmp = y.transpose() @ np.log(h) + (ones - y).transpose() @ np.log(ones - h)
	reg_term = lamda / (2 * x.shape[0]) * (theta[1:].transpose() @ theta[1:])
	j = -(tmp) / x.shape[0] + reg_term
	return j

def gradientReg(theta, x, y, lamda):
	tmp = x.transpose() @ (g(x @ theta) - y)
	reg_term = (lamda / x.shape[0]) * theta
	reg_term[0] = 0
	grad = tmp + reg_term
	return grad

def getClassifiers():
	global x, y
	lamda = 1
	#theta contains 10 classifiers, where theta[i] is the ith classifier
	theta = np.zeros((10, x.shape[1]), dtype=np.float64)
	for i in range(10):
		y_i = np.array([1 if elem == i else 0 for elem in y])
		fmin = opt.minimize(fun=J_Reg, x0=theta[i], args=(x, y_i, lamda), method='TNC', jac=gradientReg)
		theta[i] = fmin.x
	return theta

def findAccuracy(theta, x, y):
	predictor = x @ theta.transpose()
	predicted = np.argmax(predictor, axis=1)
	diff = predicted - y
	correct = 0
	for elem in diff:
		if (elem == 0):
			correct += 1
	accuracy = correct / y.shape[0]
	return accuracy

tmp = np.ones(((data['X'].shape)[0],1), dtype=np.float64)
x = np.insert(data['X'], [0], tmp, axis=1)
y_prime = data['y'].transpose()[0]
y = np.array([elem if elem != 10 else 0 for elem in y_prime])

theta = getClassifiers()
print(findAccuracy(theta, x, y))




