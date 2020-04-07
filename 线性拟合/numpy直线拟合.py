import numpy as np
import matplotlib.pyplot as plt

# x的个数决定了样本量
x = np.arange(-1, 1, 0.02)
# y为理想函数
y = 2 * np.sin(x * 2.3) + 0.5 * x ** 3
# y1为离散的拟合数据
y1 = y + 0.5 * (np.random.rand(len(x)) - 0.5)

##################################
# 主要程序
one = np.ones((len(x), 1))  # len(x)得到数据量
x = x.reshape(x.shape[0], 1)
A = np.hstack((x, one))  # 两个100x1列向量合并成100x2,(100, 1) (100,1 ) (100, 2)
b = y1.reshape(y1.shape[0], 1)


# 等同于C=y1.reshape(100,1)
# 虽然知道y1的个数为100但是程序中不应该出现人工读取的数据

def optimal(A, b):
    B = A.T.dot(b)
    AA = np.linalg.inv(A.T.dot(A))  # 求A.T.dot(A)的逆
    P = AA.dot(B)
    print(P)
    return A.dot(P)


# 求得的[a,b]=P=[[  2.88778507e+00] [ -1.40062271e-04]]
yy = optimal(A, b)
# yy=P[0]*x+P[1]
##################################
plt.plot(x, y, color='g', linestyle='-', marker='', label=u'理想曲线')
plt.plot(x, y1, color='m', linestyle='', marker='o', label=u'拟合数据')
plt.plot(x, yy, color='b', linestyle='-', marker='.', label=u"拟合曲线")
# 把拟合的曲线在这里画出来
plt.legend(loc='upper left')
plt.show()

