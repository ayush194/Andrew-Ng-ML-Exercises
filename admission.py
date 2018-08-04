import os
import numpy as np
import math
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

def J(data, theta):
	#using the vector notation
	x = data[:, :-1]
	y = data[:, -1]
	ones = np.ones((y.shape[0],), dtype=np.float64)
	h = g(x @ theta)
	tmp = y.transpose() @ np.log(h) + (ones - y).transpose() @ np.log(ones - h)
	j = -(tmp) / data.shape[0]
	return j

def regression(data):
	alpha = 0.001
	#there is a kinda flat region (encountered before the global minima) where j
	#j converges when epsilon > 0.000001
	epsilon = 0.000001
	theta = np.zeros((data.shape[1] - 1,), dtype=np.float64)
	j = J(data, theta)
	j_prev = j + 1;
	while (abs(j - j_prev) > epsilon):
		#gradient descent updates
		x = data[:, :-1]
		y = data[:, -1]
		tmp = x.transpose() @ (g(x @ theta) - y)
		theta = theta - (alpha / data.shape[0]) * tmp
		j_prev = j;
		j = J(data, theta)
		print(j)
	return theta

def abline(slope, intercept):
    #Plot a line from slope and intercept
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')

theta = regression(data)
print(theta)


plt.plot([row[1] for row in data if row[-1] == 0], [row[2] for row in data if row[-1] == 0], "yo")
plt.plot([row[1] for row in data if row[-1] == 1], [row[2] for row in data if row[-1] == 1], "k+")
abline(-theta[1] / theta[2], -theta[0] / theta[2])
plt.ylabel('admissions')
plt.show()


