import math
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

data = loadmat('data/ex3data1.mat')
theta = loadmat('data/ex4weights.mat')
theta1 = theta['Theta1']
theta2 = theta['Theta2']

x = data['X']
y_prime = data['y'].transpose()[0]
y_prime = np.array([elem if elem != 10 else 0 for elem in y_prime])
y = np.zeros((y_prime.shape[0], 10))
for i in range(y_prime.shape[0]):
	y[i][y_prime[i]] = 1

def g(z):
	#sigmoid function
	return np.float64(1.0) / (1.0 + math.pow(math.e, -z))
	
g = np.vectorize(g, otypes=[np.float64])
#now, when g is applied to a vector, it is applied to each cell

def unroll(thetas):
	unrolled_theta = np.concatenate(tuple([theta.flatten() for theta in thetas]), axis=0)
	return unrolled_theta

def roll(theta, layers):
	#returns a list of rolled theta matrices
	rolled_theta = []
	offset = 0
	for i in range(1, len(layers)):
		m, n = layers[i], layers[i - 1] + 1
		rolled_theta.append(theta[offset: offset + m * n].reshape((m, n)))
		offset += m * n
	return rolled_theta

def randomInit(layers):
	rolled_theta = []
	for i in range(1, len(layers)):
		m, n = layers[i], layers[i - 1] + 1
		rolled_theta.append((np.random.rand(m, n) - 0.5 * np.ones((m, n))))
	unrolled_theta = unroll(rolled_theta)
	return unrolled_theta

def cost(theta, h, y, layers, λ):
	#ith row of h or y is the hypothesis or expected output for the ith training set
	cost, m = 0, y.shape[0]
	for i in range(m):
		cost += y[i] @ np.log(h[i]) + (np.ones((y[i].shape)) - y[i]) @ np.log(np.ones(h[i].shape) - h[i])
	cost = -cost / m
	rolled_theta = roll(theta, layers)
	for theta_i in rolled_theta:
		cost += (λ / (2 * m)) * (np.sum(np.square(theta_i[:,1:])))
	return cost

def classifier(h):
	tmp = np.argmax(h, axis=1)
	output = np.zeros((h.shape), dtype='float64')
	for i in range(tmp.shape[0]):
		output[i][tmp[i]] = 1
	return output	

def feedForward(theta, x, layers):
	#returns a list or np.ndarrays where list[i][j] is the
	#activator vector for the ith layer and jth dataset
	rolled_theta = roll(theta, layers)
	a = [x.transpose(),]
	for i in range(len(layers) - 1):
		a[-1] = np.insert(a[-1], [0], np.ones((a[-1].shape[1],), dtype=np.float64), axis=0)
		a.append(g(rolled_theta[i] @ a[-1]))
	for i in range(len(a)):
		a[i] = a[i].transpose()
	return a
	
def gradient(theta, x, y, layers, λ):
	print("running gradient descent")
	Δ = [0 for el in range(len(layers) - 1)]
	m = x.shape[0]
	a = feedForward(theta, x, layers)
	rolled_theta = roll(theta, layers)
	#list of np.ndarrays where list[i][j] contains δ for ith layer and jth dataset
	#δ fr ith layer is a matrix of m rows, each row for one training set
	δ = [0 for el in layers]
	δ[-1] = a[-1] - y
	for i in range(len(layers) - 2, -1, -1):
		g_prime = (a[i] * (np.ones(a[i].shape) - a[i]))
		tmp = np.array([rolled_theta[i].transpose() @ row for row in δ[i + 1]])
		#no error term for i = 0, but here we are calculating just for fun!
		#also no error term for bias units which is why we discard the first column of δ[i]
		δ[i] = (tmp * g_prime)[:,1:]
		Δ[i] = sum([np.outer(δ[i + 1][j], a[i][j]) for j in range(m)])
	for i in range(len(layers) - 1):
		tmp = np.insert(rolled_theta[i][:,1:], 0, np.zeros((rolled_theta[i].shape[0],)), axis=1)
		Δ[i] = Δ[i] / np.float64(m) + (λ / np.float64(m)) * tmp
	Δ_unrolled = unroll(Δ)
	return Δ_unrolled

def grad_check(theta, x, y, layers, λ):
	epsilon = 0.0001
	Δ_unrolled = np.zeros((theta.shape), dtype=np.float64)
	for i in range(len(theta) -1, 0, -1):
		tmp = np.zeros((theta.shape))
		tmp[i] = epsilon
		theta_up = theta + tmp
		theta_lo = theta - tmp
		cost_up = cost(theta_up, feedForward(theta_up, x, layers)[-1], y, layers, λ)
		cost_lo = cost(theta_lo, feedForward(theta_lo, x, layers)[-1], y, layers, λ)
		Δ_unrolled[i] = (cost_up - cost_lo) / (2 * epsilon)
		print(i, Δ_unrolled[i])
	return Δ_unrolled

def findAccuracy(theta, x, y, layers):
	y_pred = classifier(feedForward(theta, x, layers)[-1])
	correct = [1 if np.all(a == b) else 0 for (a, b) in zip(y_pred, y)]
	accuracy = (sum(map(int, correct)) / float(len(correct)))
	return accuracy

layers = [400, 25, 10]
#theta = unroll((theta1, theta2))
#randomly initialize a theta
theta = randomInit(layers)
λ = 1
#h = feedForward(theta, x, layers)[-1]
#print(cost(theta, h, y, layers, λ))
'''
grad = gradient(theta, x, y, layers, λ)
for i in range(grad.shape[0]):
	print(i, grad[i])
print("//")
print(grad_check(theta, x, y, layers, λ))
'''
#a = np.array([[[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10], [11, 12, 13]]])
#print(a.shape)
#out = classifier(h)
#print(out[4003])
#print(y[4003])

def backPropagate(theta, x, y, layers, λ):
	j = cost(theta, feedForward(theta, x, layers)[-1], y, layers, λ)
	grad = gradient(theta, x, y, layers, λ)
	return j, grad

fmin = opt.minimize(fun=backPropagate, x0=theta, args=(x, y, layers, λ), method='TNC', jac=True, options={'maxiter': 250})
theta = fmin.x
np.set_printoptions(threshold=np.nan)
print(fmin)
print(findAccuracy(theta, x, y, layers))
