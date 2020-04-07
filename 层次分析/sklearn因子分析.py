from sklearn import datasets
iris = datasets.load_iris()
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=2)
iris_two_dim = fa.fit_transform(iris.data)
iris_two_dim[:5]

#
# array([[-1.33125848,  0.55846779],
#        [-1.33914102, -0.00509715],
#        [-1.40258715, -0.30798300],
#        [-1.29839497, -0.71854288],
#        [-1.33587575,  0.36533259]])

from matplotlib import pyplot as plt
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.scatter(iris_two_dim[:,0], iris_two_dim[:, 1], c=iris.target)
ax.set_title("Factor Analysis 2 Components")