import numpy as np
import Pandas学习 as pd
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('iris.data')
print(df.head())
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.head()
X = df.ix[:,0:4].values
y = df.ix[:,4].values
X_std = StandardScaler().fit_transform(X)
print(X_std)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

