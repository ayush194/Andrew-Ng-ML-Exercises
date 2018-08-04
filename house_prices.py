import os
import numpy as np
import matplotlib.pyplot as plt

path = os.getcwd() + "/data/ex1data2.txt"
data = np.loadtxt(path, delimiter=",")
temp = np.ones(((data.shape)[0],1), dtype=np.float64)
data = np.append(temp, data, axis=1)

def partial(data, theta, i):
	temp = 0.0
	for row in data:
		temp += (np.inner(row[:-1], theta) - row[-1]) * row[i]
	temp /= data.shape[0]
	return temp

def J(data, theta):
	j = 0.0;
	for row in data:
		#print(theta.transpose())
		#print(np.multiply(row[:-1], theta.transpose()))
		j += (np.inner(row[:-1], theta) - row[-1]) ** 2
	j /= (2 * data.shape[0])
	return j

def J_alt(data, theta):
	#an easier vector implementation of J
	x = data[:, :-1]
	y = data[:, -1]
	tmp = x @ theta - y
	j = (tmp.transpose() @ tmp) / (2 * data.shape[0])
	return j

def regression(data):
	#for plotting error vs training epoch
	#count = 0
	alpha = 0.1
	#np.zeros gives a 2d array which we need to convert to 1d
	theta = np.zeros((1, (data.shape)[1] - 1))[0]
	j = J(data, theta)
	j_prev = j + 1
	epsilon = 0.0001
	while (abs(j - j_prev) >= epsilon):
		theta_temp = np.zeros((1, data.shape[1] - 1))[0]
		'''
		for i in range(data.shape[1] - 1):
			theta_temp[i] = alpha * partial(data, theta, i)
		theta = np.subtract(theta, theta_temp)
		'''
		#vectorized implementation of finding gradient
		x = data[:, :-1]
		y = data[:, -1]
		tmp = x.transpose() @ (x @ theta - y)
		theta = theta - (alpha / data.shape[0]) * tmp
		j_prev = j
		j = J_alt(data, theta)
		#print(j)
		#plt.plot(count, j, "b.")
		#count += 1
	return theta

def featureScaling(data):
	mean = np.zeros((1, data.shape[1] - 1))[0]
	min_ = data[0][:-1]
	max_ = data[0][:-1]
	for row in data:
		mean = np.add(mean, row[:-1])
		min_ = np.minimum(min_, row[:-1])
		max_ = np.maximum(max_, row[:-1])
	for j in range(data.shape[1] - 1):
		mean[j] /= data.shape[0]
	for i in range(data.shape[0]):
		for j in range(1, data.shape[1] - 1):
			data[i][j] = (data[i][j] - mean[j]) / (max_[j] - min_[j])

def abline(slope, intercept):
    #Plot a line from slope and intercept
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')


featureScaling(data)
#print(data)
theta = regression(data)
print(theta)

'''
plt.plot(data[:,1], data[:,2], "r.")
abline(theta[1], theta[0])
plt.ylabel('some numbers')
plt.show()
'''