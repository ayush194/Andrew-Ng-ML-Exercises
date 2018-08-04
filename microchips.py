import os
import pandas as pd
import math
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

path = os.getcwd() + '/data/ex2data2.txt'  
data = np.loadtxt(path, delimiter=",")
temp = np.ones(((data.shape)[0],1), dtype=np.float64)
data = np.insert(data, [0], temp, axis=1)

def addParams(data):
	new_params = np.zeros((data.shape[0],25), dtype=np.float64)
	count = 0
	for i in range(2, 7):
		for j in range(i + 1):
			x1 = data[:,1]
			x2 = data[:,2]
			new_params[:, count] = np.power(x1, i - j) * np.power(x2, j)
			count += 1
	data = np.insert(data, [3], new_params, axis=1)
	return data

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

data = addParams(data)
x = data[:, :-1]
y = data[:, -1]
theta = np.zeros((data.shape[1] - 1,), dtype=np.float64)
lamda = 1
#print(J_Reg(theta, x, y, lamda))
result = opt.fmin_tnc(func=J_Reg, x0=theta, fprime=gradientReg, args=(x, y, lamda))
theta = result[0]


delta = 0.025
xrange = np.arange(-5.0, 20.0, delta)
yrange = np.arange(-5.0, 20.0, delta)
X1, X2 = np.meshgrid(xrange,yrange)

# F is one side of the equation, G is the other
F = 0
count = 0
for i in range(7):
	for j in range(i + 1):
		F += theta[count] * np.power(X1, i - j) * np.power(X2, j)
		count += 1
'''
F = theta[0] + theta[1]*X1 + theta[2]*X2 + theta[3]*X1*X1 + theta[4]*X1*X2 + theta[5]*X2*X2 + \
theta[6]*X1*X1*X1 + theta[7]*X1*X1*X2 + theta[8]*X1*X2*X2 + theta[9]*X2*X2*X2 + \
theta[10]*X1*X1*X1*X1 + theta[11]*X1*X1*X1*X2 + theta[12]*X1*X1*X2*X2 + theta[13]*X1*X2*X2*X2 + theta[14]*X2*X2*X2*X2 + \
theta[15]*X1*X1*X1*X1*X1 + theta[16]*X1*X1*X1*X1*X2 + theta[17]*X1*X1*X1*X2*X2 + theta[18]*X1*X1*X2*X2*X2 + theta[19]*X1*X2*X2*X2*X2 + theta[20]*X2*X2*X2*X2*X2 + \
theta[21]*X1*X1*X1*X1*X1*X1 + theta[22]*X1*X1*X1*X1*X1*X2 + theta[23]*X1*X1*X1*X1*X2*X2 + theta[24]*X1*X1*X1*X2*X2*X2 + theta[25]*X1*X1*X2*X2*X2*X2 + theta[26]*X1*X2*X2*X2*X2*X2 + theta[27]*X2*X2*X2*X2*X2*X2
'''

plt.contour(X1, X2, (F), [0], colors=["k"])
#x1 on x-axis and x2 on y-axis
plt.plot([row[1] for row in data if row[-1] == 0], [row[2] for row in data if row[-1] == 0], "yo")
plt.plot([row[1] for row in data if row[-1] == 1], [row[2] for row in data if row[-1] == 1], "k+")
plt.ylabel('admissions')
plt.show()


'''
positive = data[data['Accepted'].isin([1])]  
negative = data[data['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')  
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')  
ax.legend()  
ax.set_xlabel('Test 1 Score')  
ax.set_ylabel('Test 2 Score')
plt.show()
'''