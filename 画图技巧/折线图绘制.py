import pandas as pd
import matplotlib.pyplot as plt


# 读数据
unrate = pd.read_csv('unrate.csv')
# 类型转换DATE到datetime格式
unrate['DATE'] = pd.to_datetime(unrate['DATE'])
print(unrate.head(12))
# 画图
plt.plot()# 画图操作
plt.show()# 显示操作

# 画出内容
first_twelve = unrate[0:12]
plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
plt.show()
# .xticks改变坐标变换45°
plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
plt.xticks(rotation=45)
# print help(plt.xticks)
plt.show()
# 加上一些条件
plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
plt.xticks(rotation=90)
plt.xlabel('Month')
plt.ylabel('Unemployment Rate')
plt.title('Monthly Unemployment Trends, 1948')
plt.show()
