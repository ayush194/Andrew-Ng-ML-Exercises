import math
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn import svm

raw_data = loadmat('data/ex6data1.mat')
x = raw_data['X']
y = raw_data['y'][:,0]
m = x.shape[0]

#vary C to see the changes in the decision boundary
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(x, y)
print(svc.score(x, y))

fig, ax = plt.subplots(figsize=(12,8))  
#ax.scatter([x[i][0] for i in range(m) if y[i] == 1], [x[i][1] for i in range(m) if y[i] == 1], s=50, marker='x', label='Positive')  
#ax.scatter([x[i][0] for i in range(m) if y[i] == 0], [x[i][1] for i in range(m) if y[i] == 0], s=50, marker='o', label='Negative')  
#ax.legend()

ax.scatter(x[:,0], x[:,1], s=50, c=svc.decision_function(x), cmap='seismic')  
ax.set_title('SVM (C=1) Decision Confidence')
#plt.plot([x[i][0] for i in range(m) if y[i] == 0], [x[i][1] for i in range(m) if y[i] == 0], 'yo')
#plt.plot([x[i][0] for i in range(m) if y[i] == 1], [x[i][1] for i in range(m) if y[i] == 1], 'kx')
plt.show()