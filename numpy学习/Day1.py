# 1.1 列表转为矩阵
import numpy as np
from numpy import array

array = np.array(
    [[1, 3, 5],
     [4, 6, 9]]
)
print(array)
# 1.2 维度
print('number of dim:', array.ndim)
# 1.3行数和列数的和
print('shape:', array.shape)
# 1.4 元素个数
print('size:', array.size)
# 2.Numpy创建array
# 2.1 一维array创建
a = np.array([2, 23, 4], dtype=np.int32) # np.int默认为int32
print(a)
print(a.dtype)
# 2.2 多维array创建
a = np.array([[2, 3, 4],
              [3, 4, 5]])
print(a)  # 生成2行3列的矩阵
# 2.3 创建全零数组
a = np.zeros((3, 4))
print(a)  # 生成3行4列的全零矩阵
# 2.4 创建全1数据
# 创建全一数据，同时指定数据类型
a = np.ones((3,4),dtype=np.int)
print(a)
# 2.5 创建全空数组
# 创建全空数组，其实每个值都是接近于零的数
a = np.empty((3,4))
print(a)
# 2.6 创建连续数组
# 创建连续数组
a = np.arange(10,21,2)  # 10-20的数据，步长为2
print(a)
# 2.7 reshape操作
# 使用reshape改变上述数据的形状
b = a.reshape((2,3))
print(b)
# 2.8 创建连续型数据
# 创建线段型数据
a = np.linspace(1,10,20) # 开始端1，结束端10，且分割成20个数据，生成线段
print(a)
# 2.9 linspace的reshape操作
# 同时也可以reshape
b = a.reshape((5,4))
print(b)
# 3.Numpy基本运算
# 一维矩阵运算
a = np.array([10,20,30,40])
b = np.arange(4)
print(a, b)
c = a - b
print(c)
print(a*b)  # 若用a.dot(b),则为各维之和
# 在Numpy中，想要求出矩阵中各个元素的乘方需要依赖双星符号 **，以二次方举例，即：
c = b**2
print(c)
# Numpy中具有很多的数学函数工具
c = np.sin(a)
print(c)
print(b < 2)
a = np.array([1,1,4,3])
b = np.arange(4)
print(a == b)
# 3.2 多维矩阵运算
a = np.array([[1,1],[0,1]])
b = np.arange(4).reshape((2,2))
print(a)
print(b)
# 多维度矩阵乘法
# 第一种乘法方式:
c = a.dot(b)
print(c)
# 第二种乘法:
c = np.dot(a,b)
print(c)
# 多维矩阵乘法不能直接使用'*'号
a = np.random.random((2, 4))
# 如果你需要对行或者列进行查找运算，

# 就需要在上述代码中为 axis 进行赋值。

# 当axis的值为0的时候，将会以列作为查找单元，

# 当axis的值为1的时候，将会以行作为查找单元。

print(np.sum(a))
print(np.min(a))
print(np.max(a))
print("a=", a)
print("sum=",np.sum(a,axis=1))
print("min=",np.min(a,axis=0))
print("max=",np.max(a,axis=1))
# 3.3 基本计算
A = np.arange(2,14).reshape((3,4))
print(A)
# 最小元素索引
print(np.argmin(A))  # 0
# 最大元素索引
print(np.argmax(A))  # 11
# 求整个矩阵的均值
print(np.mean(A))  # 7.5
print(np.average(A))  # 7.5
print(A.mean())  # 7.5
# 中位数
print(np.median(A))  # 7.5
# 累加
print(np.cumsum(A))
# 累差运算
B = np.array([[3,5,9],
              [4,8,10]])
print(np.diff(B))
C = np.array([[0,5,9],
              [4,0,10]])
print(np.nonzero(B))
print(np.nonzero(C))
# 仿照列表排序
A = np.arange(14,2,-1).reshape((3,4)) # -1表示反向递减一个步长
print(A)
print(np.sort(A))
# 矩阵转置
print(np.transpose(A))
print(A.T)
print(A)
print(np.clip(A,5,9))
# clip(Array,Array_min,Array_max)
#
# 将Array_min<X<Array_max X表示矩阵A中的数，如果满足上述关系，则原数不变。
#
# 否则，如果X<Array_min，则将矩阵中X变为Array_min;
#
# 如果X>Array_max，则将矩阵中X变为Array_max
# 4.Numpy索引与切片
A = np.arange(3,15)
print(A)
print(A[3])
B = A.reshape(3,4)
print(B)
print(B[2])
print(B[0][2])
print(B[0,2])
# list切片操作
print(B[1,1:3]) # [8 9] 1:3表示1-2不包含3
for row in B:
    print(row)
# 如果要打印列，则进行转置即可
for column in B.T:
    print(column)
# 多维转一维
A = np.arange(3, 15).reshape((3, 4))
# print(A)
print(A.flatten())
# flat是一个迭代器，本身是一个object属性
for item in A.flat:
    print(item)
# 5.Numpy array合并
# 5.1 数组合并
A = np.array([1,1,1])
B = np.array([2,2,2])
print(np.vstack((A,B)))
# vertical stack 上下合并,对括号的两个整体操作。
C = np.vstack((A,B))
print(C)
print(A.shape,B.shape,C.shape)# 从shape中看出A,B均为拥有3项的数组(数列)
# horizontal stack左右合并
D = np.hstack((A,B))
print(D)
print(A.shape,B.shape,D.shape)
# (3,) (3,) (6,)
# 对于A,B这种，为数组或数列，无法进行转置，需要借助其他函数进行转置
# 5.2 数组转置为矩阵
print(A[np.newaxis,:])  # [1 1 1]变为[[1 1 1]]
print(A[np.newaxis,:].shape) # (3,)变为(1, 3)
print(A[:,np.newaxis])
# 多个矩阵合并
# concatenate的第一个例子
print("------------")
print(A[:, np.newaxis].shape) # (3,1)
A = A[:,np.newaxis] # 数组转为矩阵
B = B[:,np.newaxis] # 数组转为矩阵
print(A)
print(B)
# axis=0纵向合并
C = np.concatenate((A, B, B, A), axis=0)
print(C)
# axis=1横向合并
C = np.concatenate((A, B), axis=1)
print(C)
# 5.4 合并例子2
# concatenate的第二个例子
print("-------------")
a = np.arange(8).reshape(2,4)
b = np.arange(8).reshape(2,4)
print(a)
print(b)
print("-------------")
# axis=0多个矩阵纵向合并
c = np.concatenate((a,b), axis=0)
print(c)
# axis=1多个矩阵横向合并
c = np.concatenate((a, b), axis=1)
print(c)
# 6.Numpy array分割
# 6.1 构造3行4列矩阵
A = np.arange(12).reshape((3,4))
print(A)
# 6.2 等量分割
# 等量分割
# 纵向分割同横向合并的axis
print(np.split(A, 2, axis=1))
# 横向分割同纵向合并的axis
print(np.split(A,3,axis=0))
# 6.3 不等量分割
print(np.array_split(A,3,axis=1))
# 6.4 其他的分割方式
# 横向分割
print(np.vsplit(A,3)) # 等价于print(np.split(A,3,axis=0))
# 纵向分割
print(np.hsplit(A,2)) # 等价于print(np.split(A,2,axis=1))
# `=`赋值方式会带有关联性
a = np.arange(4)
print(a)  # [0 1 2 3]
b = a
c = a
d = b
a[0] = 11
print(a)  # [11  1  2  3]
print(b)  # [11  1  2  3]
print(c)  # [11  1  2  3]
print(d)  # [11  1  2  3]
print(b is a)  # True
print(c is a)  # True
print(d is a)  # True
d[1:3] = [22,33]
print(a)  # [11 22 33  3]
print(b)  # [11 22 33  3]
print(c)  # [11 22 33  3]
# 7.2 copy()赋值方式没有关联性
a = np.arange(4)
print(a)  # [0 1 2 3]
b =a.copy()  # deep copy
print(b)  # [0 1 2 3]
a[3] = 44
print(a) # [ 0  1  2 44]
print(b) # [0 1 2 3]
# 9.常用函数
# 9.1 np.bincount()
x = np.array([1, 2, 3, 3, 0, 1, 4])
print(np.bincount(x))
# 统计索引出现次数：索引0出现1次，1出现2次，2出现1次，3出现2次，4出现1次
#
# 因此通过bincount计算出索引出现次数如下：
#
# 上面怎么得到的？
#
# 对于bincount计算吗，bin的数量比x中最大数多1，例如x最大为4，那么bin数量为5(index从0到4)，也就会bincount输出的一维数组为5个数，bincount中的数又代表什么？代表的是它的索引值在x中出现的次数！
w = np.array([0.3,0.5,0.7,0.6,0.1,-0.9,1])
print(np.bincount(x,weights=w))
# 9.2 np.argmax()
# 函数原型为：numpy.argmax(a, axis=None, out=None).
# 函数表示返回沿轴axis最大值的索引。
x = [[1,3,3],
     [7,5,2]]
print(np.argmax(x))
x = [[1,3,3],
     [7,5,2]]
print(np.argmax(x,axis=0))
# 9.3 上述合并实例
x = np.array([1, 2, 3, 3, 0, 1, 4])
print(np.argmax(np.bincount(x)))
# 9.4 求取精度
np.around([-0.6,1.2798,2.357,9.67,13], decimals=0)#取指定位置的精度









