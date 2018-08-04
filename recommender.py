import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.io import loadmat
from scipy.optimize import minimize

data = loadmat('data/ex8_movies.mat')
#params_data = loadmat('data/ex8_movieParams.mat')
y, r = data['Y'], data['R']

def randomInit(mov, us, feat):
	x = np.random.rand(mov, feat)
	theta = np.random.rand(us, feat)
	params = np.concatenate((np.ravel(x), np.ravel(theta)))
	return params

def gradient(params, y, r, n_feat, λ):
	nm = y.shape[0]
	nu = y.shape[1]
	#unravel
	x = params[:n_feat * nm].reshape((nm, n_feat))
	theta = params[n_feat * nm:].reshape((nu, n_feat))
	x_new = ((x @ theta.T - y) * r) @ theta
	theta_new = ((x @ theta.T - y) * r).T @ x
	#regularisation
	x_new += λ * x
	theta_new += λ * theta
	#ravel
	params_new = np.concatenate((x_new.ravel(), theta_new.ravel()))
	return params_new

def J(params, y, r, feat, λ):
	nm = y.shape[0]
	nu = y.shape[1]
	#unravel
	x = params[:feat * nm].reshape((nm, feat))
	theta = params[feat * nm:].reshape((nu, feat))
	tmp = (x @ theta.T - y) * r
	j = np.power(tmp, 2).sum() / 2
	#regularisation
	j += (np.float64(λ) / 2) * np.power(x, 2).sum()
	j += (np.float64(λ) / 2) * np.power(theta, 2).sum()
	grad = gradient(params, y, r, feat, λ)
	return j, grad

#for checking i the cost function works correctly
#mov, us, feat = 5, 4, 3
#x_check, theta_check = params_data['X'], params_data['Theta']
#params_check = np.concatenate((x_check[:mov,:feat].ravel(), theta_check[:us,:feat].ravel()))
#j, grad = J(params_check, y[:mov,:us], r[:mov,:us], feat, 1.5)
#print(j, grad)
feat, λ = 100, 1
params = randomInit(y.shape[0], y.shape[1], feat)
fmin = minimize(fun=J, x0=params, args=(y, r, feat, λ), method='CG', jac=True, options={'maxiter': 100})
print(fmin)

fig, ax = plt.subplots(figsize=(12,8))
ax.imshow(y)
plt.show()