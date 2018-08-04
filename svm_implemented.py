import math
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn import svm

raw_data = loadmat('data/ex6data2.mat')
x = raw_data['X']
y = raw_data['y'][:,0]
m = x.shape[0]

def gaussian(xi, xj, σ):
	tmp = xi - xj
	norm = (np.inner(tmp, tmp)).sum()
	k = math.exp(-norm / np.float64(2 * (σ ** 2)))
	return k

#print(gaussian(y, y, 1))

svc = svm.SVC(C=100, gamma=10, probability=True, kernel=gaussian)
svc.fit(x, y)

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(x[:,0], x[:,1], s=30, c=svc.predict_proba(x)[:,0], cmap='Reds') 

#positive = np.array([x[i] for i in range(m) if y[i] == 1])
#negative = np.array([x[i] for i in range(m) if y[i] == 0]) 
#ax.scatter(positive[:,0], positive[:,1], s=30, marker='x', label='Positive')  
#ax.scatter(negative[:,0], negative[:,1], s=30, marker='o', label='Negative')  
#ax.legend()
plt.show()