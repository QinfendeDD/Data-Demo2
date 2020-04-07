# 首先，训练集有6组数据，每组数据有4个特征，我们的目的是将其降到2维，也就是2个特征。首先，训练集有6组数据，每组数据有4个特征，我们的目的是将其降到2维，也就是2个特征。
#coding=utf-8
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
pca = PCA(n_components=2)   # 降到2维
pca.fit(X)                  # 训练
newX=pca.fit_transform(X)   # 降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  # 输出贡献率
print(newX)                  # 输出降维后的数据
# 参数注释
# n_components:  我们可以利用此参数设置想要的特征维度数目，可以是int型的数字，也可以是阈值百分比，如95%，让PCA类根据样本特征方差来降到合适的维数，也可以指定为string类型，MLE。
# copy： bool类型，TRUE或者FALSE，是否将原始数据复制一份，这样运行后原始数据值不会改变，默认为TRUE。
# whiten：bool类型，是否进行白化（就是对降维后的数据进行归一化，使方差为1），默认为FALSE。如果需要后续处理可以改为TRUE。
# explained_variance_： 代表降为后各主成分的方差值，方差值越大，表明越重要。
# explained_variance_ratio_： 代表各主成分的贡献率。
# inverse_transform()： 将降维后的数据转换成原始数据，X=pca.inverse_transform(newX)。
