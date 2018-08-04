import os
import numpy as np
import matplotlib.pyplot as plt

path = os.getcwd() + "/data/ex1data2.txt"
data = np.loadtxt(path, delimiter=",")
temp = np.ones(((data.shape)[0],1), dtype=np.float64)
data = np.append(temp, data, axis=1)

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


def regression(data):
	#using normal equations method
	x = data[:, :-1]
	y = data[:, -1]
	return  (np.linalg.inv(x.transpose() @ x) @ x.transpose()) @ y

#no need to apply feature scaling when we are using normal equations method
#featureScaling(data)
print(regression(data))
