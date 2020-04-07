import Pandas学习 as pd
import numpy as np
import matplotlib.pyplot as plt
from Pandas学习 import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 创建数据集
examDict = {'学习时间': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75,
                     2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
            '分数': [10, 22, 13, 43, 20, 22, 33, 50, 62,
                   48, 55, 75, 62, 73, 81, 76, 64, 82, 90, 93]}

# 转换为DataFrame的数据格式
examDf = DataFrame(examDict)

# 绘制散点图
plt.scatter(examDf.分数, examDf.学习时间, color='b', label="Exam Data")

# 添加图的标签（x轴，y轴）
plt.xlabel("Hours")
plt.ylabel("Score")
# 显示图像
plt.savefig("examDf.jpg")
plt.show()

# 相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
# 相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
rDf = examDf.corr()
print(rDf)

# 回归方程 y = a + b*x (模型建立最佳拟合线)
# 点误差 = 实际值 - 拟合值
# 误差平方和（Sum of square error） SSE = Σ（实际值-预测值）^2
# 最小二乘法 ： 使得误差平方和最小（最佳拟合）
exam_X = examDf.loc[:, '学习时间']
exam_Y = examDf.loc[:, '分数']

# 将原数据集拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(exam_X, exam_Y, train_size=.8)
# X_train为训练数据标签,X_test为测试数据标签,exam_X为样本特征,exam_y为样本标签，train_size 训练数据占比

print("原始数据特征:", exam_X.shape,
      ",训练数据特征:", X_train.shape,
      ",测试数据特征:", X_test.shape)

print("原始数据标签:", exam_Y.shape,
      ",训练数据标签:", Y_train.shape,
      ",测试数据标签:", Y_test.shape)

# 散点图
plt.scatter(X_train, Y_train, color="blue", label="train data")
plt.scatter(X_test, Y_test, color="red", label="test data")

# 添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Pass")
# 显示图像
plt.savefig("tests.jpg")
plt.show()

model = LinearRegression()

# 对于下面的模型错误我们需要把我们的训练集进行reshape操作来达到函数所需要的要求
# model.fit(X_train,Y_train)

# reshape如果行数=-1的话可以使我们的数组所改的列数自动按照数组的大小形成新的数组
# 因为model需要二维的数组来进行拟合但是这里只有一个特征所以需要reshape来转换为二维数组
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

model.fit(X_train, Y_train)

a = model.intercept_  # 截距

b = model.coef_  # 回归系数

print("最佳拟合线:截距", a, ",回归系数：", b)

# 决定系数r平方
# 对于评估模型的精确度
# y误差平方和 = Σ(y实际值 - y预测值)^2
# y的总波动 = Σ(y实际值 - y平均值)^2
# 有多少百分比的y波动没有被回归拟合线所描述 = SSE/总波动
# 有多少百分比的y波动被回归线描述 = 1 - SSE/总波动 = 决定系数R平方
# 对于决定系数R平方来说1） 回归线拟合程度：有多少百分比的y波动刻印有回归线来描述(x的波动变化)
# 2）值大小：R平方越高，回归模型越精确(取值范围0~1)，1无误差，0无法完成拟合

plt.scatter(X_train, Y_train, color='blue', label="train data")

# 训练数据的预测值
y_train_pred = model.predict(X_train)
# 绘制最佳拟合线：标签用的是训练数据的预测值y_train_pred
plt.plot(X_train, y_train_pred, color='black', linewidth=3, label="best line")

# 测试数据散点图
plt.scatter(X_test, Y_test, color='red', label="test data")

# 添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")
# 显示图像
plt.savefig("lines.jpg")
plt.show()

score = model.score(X_test, Y_test)

print(score)