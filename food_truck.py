import os  
import numpy as np   
import matplotlib.pyplot as plt
import pylab
#%matplotlib inline

path = os.getcwd() + "/data/ex1data1.txt"
data = np.loadtxt(path, delimiter=",")
temp = np.ones((97,1), dtype=np.float64)
data = np.append(temp, data, axis=1)
'''
print(data.itemsize)
print(data.ndim)
print(data.shape)
print(data.size)
'''

def partial(data, theta0, theta1, i):
	temp = 0.0
	for row in data:
		temp += (theta0 * row[0] + theta1 * row[1] - row[2]) * row[i]
	temp /= (data.shape)[0]
	return temp

def J(theta0, theta1, data):
	j = 0.0;
	for row in data:
		j += (theta0 * row[0] + theta1 * row[1] - row[2]) ** 2
	j /= (2 * (data.shape)[0])
	return j

def regression(data):
	#for plotting error vs training epoch
	#count = 0
	alpha = 0.01
	m, n = (data.shape)[0], (data.shape)[1]
	theta0, theta1 = 0.0, 0.0
	j = J(theta0, theta1, data)
	j_prev = j + 1
	epsilon = 0.0001
	while (abs(j - j_prev) >= epsilon):
		temp0 = theta0 - alpha * partial(data, theta0, theta1, 0)
		temp1 = theta1 - alpha * partial(data, theta0, theta1, 1)
		theta0 = temp0
		theta1 = temp1
		j_prev = j
		j = J(theta0, theta1, data)
		#plt.plot(count, j, "b.")
		#count += 1
	return theta0, theta1

def abline(slope, intercept):
    #Plot a line from slope and intercept
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')

theta0, theta1 = regression(data)
temp1 = [theta0, theta0 + theta1]

plt.plot(data[:,1], data[:,2], "r.")
abline(theta1, theta0)
plt.ylabel('some numbers')
plt.show()