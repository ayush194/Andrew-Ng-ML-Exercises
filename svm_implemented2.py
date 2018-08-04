import math
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn import svm

raw_data = loadmat('data/ex6data3.mat')
x_train = raw_data['X']
y_train = raw_data['y'][:,0]
x_val = raw_data['Xval']
y_val = raw_data['yval'][:,0]

c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
σ_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score, best_c, best_σ = 0, 0, 0
for c in c_values:
	for σ in σ_values:
		svc = svm.SVC(C=c, gamma=σ)
		svc.fit(x_train, y_train)
		score = svc.score(x_val, y_val)
		if (score > best_score):
			best_score , best_c, best_σ = score, c, σ

print(best_score, best_c, best_σ)

svc = svm.SVC(C=c, gamma=σ, probability=True)
svc.fit(x_train, y_train)

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(x_val[:,0], x_val[:,1], s=30, c=svc.predict_proba(x_val)[:,0], cmap='Reds') 

#positive = np.array([x[i] for i in range(m) if y[i] == 1])
#negative = np.array([x[i] for i in range(m) if y[i] == 0]) 
#ax.scatter(positive[:,0], positive[:,1], s=30, marker='x', label='Positive')  
#ax.scatter(negative[:,0], negative[:,1], s=30, marker='o', label='Negative')  
#ax.legend()
plt.show()