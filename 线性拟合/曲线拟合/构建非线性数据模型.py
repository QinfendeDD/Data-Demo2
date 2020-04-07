import keras
import numpy as np
import matplotlib.pyplot as plt
# Sequential按顺序构成的模型
from keras.models import Sequential
# Dense全连接层
from keras.layers import Dense, Activation
# 优化器：随机梯度下降
from keras.optimizers import SGD

# 生成非线性数据模型
x_data = np.linspace(-0.5,0.5,200)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 显示随机点
plt.scatter(x_data, y_data)
plt.show()
