import os
import numpy as np
import math
import scipy.optimize as opt
import matplotlib.pyplot as plt

path = os.getcwd() + "/data/ex2data1.txt"
data = np.loadtxt(path, delimiter=",")
temp = np.ones(((data.shape)[0],1), dtype=np.float64)
data = np.append(temp, data, axis=1)

def g(z):
	#sigmoid function
	return 1.0 / (1 + math.pow(math.e, -z))
	
g = np.vectorize(g, otypes=[np.float64])
#now, when g is applied to a vector, it is applied to each cell

def J(theta, x, y):
	#using the vector notation
	ones = np.ones((y.shape[0],), dtype=np.float64)
	h = g(x @ theta)
	tmp = y.transpose() @ np.log(h) + (ones - y).transpose() @ np.log(ones - h)
	j = -(tmp) / data.shape[0]
	return j

def gradient(theta, x, y):
		tmp = x.transpose() @ (g(x @ theta) - y)
		return tmp

def findAccuracy(theta, data):
	predicted = data[:, :-1] @ theta
	for i in range(predicted.shape[0]):
		if (predicted[i] >= 0.5):
			predicted[i] = 1
		else:
			predicted[i] = 0
	diff = predicted - data[:, -1]
	error = diff.transpose() @ diff
	accuracy = (data.shape[0] - error) / data.shape[0]
	return accuracy

def abline(slope, intercept):
    #Plot a line from slope and intercept
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')

x = data[:, :-1]
y = data[:, -1]
theta = np.zeros((data.shape[1] - 1,), dtype=np.float64)
result = opt.fmin_tnc(func=J, x0=theta, fprime=gradient, args=(x, y))
print(J(result[0], x, y))
print(findAccuracy(result[0], data))

plt.plot([row[1] for row in data if row[-1] == 0], [row[2] for row in data if row[-1] == 0], "yo")
plt.plot([row[1] for row in data if row[-1] == 1], [row[2] for row in data if row[-1] == 1], "k+")
abline(-result[0][1] / result[0][2], -result[0][0] / result[0][2])
plt.ylabel('admissions')
plt.show()
