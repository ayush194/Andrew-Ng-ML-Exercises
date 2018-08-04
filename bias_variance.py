import math
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

data = loadmat('data/ex5data1.mat')
x_train = data['X']
y_train = data['y']
x_val = data['Xval']
y_val = data['yval']
x_test = data['Xtest']
y_test = data['ytest']
#print(x_train.shape)
#print(y_train.shape)
#print(x_val.shape)
#print(y_val.shape)
#print(x_test.shape)
#print(y_test.shape)

x_train = np.insert(x_train, 0, np.ones(x_train.shape[0],), axis=1)
x_val = np.insert(x_val, 0, np.ones(x_val.shape[0],), axis=1)
x_test = np.insert(x_test, 0, np.ones(x_test.shape[0],), axis=1)
y_train = y_train[:,0]
y_val = y_val[:,0]
y_test = y_test[:,0]
#print(x_train)

def J(theta, x, y, λ):
	m = x.shape[0]
	tmp = (x @ theta) - y
	j = (tmp @ tmp) / np.float64(2 * m) + (λ / np.float64(2 * m)) * (theta[1:] @ theta[1:])
	return j

def gradient(theta, x, y, λ):
	m = x.shape[0]
	tmp = (x @ theta - y)
	grad = (x.transpose() @ tmp) / np.float64(m)
	tmp = (λ / np.float64(m)) * theta
	tmp[0] = 0
	grad += tmp
	return grad

def plotLine(slope, intercept):
    #Plot a line from slope and intercept
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')

def learningCurveData(x_train, y_train, x_val, y_val, λ):
	m = x_train.shape[0]
	n = x_train.shape[1]
	theta = np.zeros((n,))
	training_error = np.zeros(m - 1)
	validation_error = np.zeros(m - 1)
	for i in range(1, m):
		x = x_train[:i,:]
		y = y_train[:i]
		fmin = opt.fmin_tnc(func=J, x0=theta, fprime=gradient, args=(x, y, λ))
		training_error[i - 1] = J(fmin[0], x, y, 0)
		validation_error[i - 1] = J(fmin[0], x_val, y_val, 0)
	return training_error, validation_error

def polyFeatures(x, p):
	m = x.shape[0]
	x_poly = np.zeros((m, p + 1))
	for i in range(p + 1):
		x_poly[:,i] = np.power(x, i)
	return x_poly

def featureScaling(data):
	m = data.shape[0]
	mean = data.sum(axis=0) / m
	std = data.std(axis=0)
	mean[0], std[0] = 0.0, 1.0
	normalized_data = (data - mean) / std
	return normalized_data, mean, std

def plotLearntFunc(theta, mean, std):
	delta = 0.25
	xrange = np.arange(-100.0, 100.0, delta)
	yrange = np.arange(-100.0, 100.0, delta)
	X, Y = np.meshgrid(xrange,yrange)
	print(X.shape)
	# F is one side of the equation, G is the other
	tmp = [0.0 for el in range(p + 1)]
	lhs = theta[0]
	for i in range(1, p + 1):
		tmp[i] = (np.power(X, i) - mean[i]) / std[i]
		lhs += theta[i] * tmp[i]
	F = lhs
	G = Y
	plt.contour(X, Y, (F - G), [0], colors=["k"])

def plotLearningCurve(x_train, y_train, x_val, y_val, λ):
	training_error, validation_error = learningCurveData(x_train, y_train, x_val, y_val, λ)
	plt.plot(training_error, 'b')
	plt.plot(validation_error, 'g')

p = 8
theta = np.zeros((p + 1,), dtype=np.float64)
#try different values of lamda to see how the learning curves vary
λ = 0
x_train_poly = polyFeatures(x_train[:,1], p)
x_val_poly = polyFeatures(x_val[:,1], p)
#mean-normalize x_train_poly & x_val_poly
x_train_poly_norm, mean_train, std_train = featureScaling(x_train_poly)
x_val_poly_norm = (x_val_poly - mean_train) / std_train

fmin = opt.fmin_tnc(func=J, x0=theta, fprime=gradient, args=(x_train_poly_norm, y_train, λ))
theta = fmin[0]

plt.plot(x_train[:,1], y_train,'rx')
plotLearntFunc(theta, mean_train, std_train)	
#plotLearningCurve(x_train_poly_norm, y_train, x_val_poly_norm, y_val, λ)
#plotLine(fmin[0][1], fmin[0][0])
plt.show()