# fig.add_subplot(2,2,x)中（2，2）是矩阵
import matplotlib.pyplot as plt
import numpy as np
import Pandas学习 as pd
fig = plt.figure()
# 定义画图域 未指定参数就画不出来
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,4)
plt.show()

# 不指定归类区，指定画图区域大小
fig = plt.figure()
fig = plt.figure(figsize=(3, 3))  # figsize指定画图区域大小
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
# 分别进行操作需要ax1.plot和ax2.plot
ax1.plot(np.random.randint(1, 5, 5), np.arange(5))
ax2.plot(np.arange(10)*3, np.arange(10))
plt.show()

# 同一个图画两条折线
# 读数据
unrate = pd.read_csv('unrate.csv')
# 类型转换DATE到datetime格式
unrate['DATE'] = pd.to_datetime(unrate['DATE'])
print(unrate.head(12))
unrate['MONTH'] = unrate['DATE'].dt.month
unrate['MONTH'] = unrate['DATE'].dt.month
fig = plt.figure(figsize=(6,3))

plt.plot(unrate[0:12]['MONTH'], unrate[0:12]['VALUE'], c='red')
plt.plot(unrate[12:24]['MONTH'], unrate[12:24]['VALUE'], c='blue')

plt.show()

# 画出所有数据的图像
fig = plt.figure(figsize=(10,6))
colors = ['red', 'blue', 'green', 'orange', 'black']
for i in range(5):
    start_index = i*12
    end_index = (i+1)*12
    subset = unrate[start_index:end_index]  # 加上注解
    label = str(1948 + i)
    plt.plot(subset['MONTH'], subset['VALUE'], c=colors[i], label=label)
plt.legend(loc='upper left')
plt.xlabel('Month, Integer')
plt.ylabel('Unemployment Rate, Percent')
plt.title('Monthly Unemployment Trends, 1948-1952')
plt.show()