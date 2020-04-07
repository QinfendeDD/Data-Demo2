import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
s = pd.Series([1,3,6,np.nan,44,1])
print(s)
# DataFrame
dates = pd.date_range('2018-08-19',periods=6)
# dates = pd.date_range('2018-08-19','2018-08-24') # 起始、结束  与上述等价
'''
numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。
numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。
(6,4)表示6行4列数据
'''
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
print(df)
# DataFrame既有行索引也有列索引， 它可以被看做由Series组成的大字典。
print(df['b'])
df1 = pd.DataFrame(np.arange(12).resize(3,4))
print(df1)
# 另一种方式
df2 = pd.DataFrame({
    'A': [1,2,3,4],
    'B': pd.Timestamp('20180819'),
    'C': pd.Series([1,6,9,10],dtype='float32'),
    'D': np.array([3] * 4,dtype='int32'),
    'E': pd.Categorical(['test','train','test','train']),
    'F': 'foo'
})
print(df2)
print(df2.index)
print(df2.columns)
print(df2.values)
# 数据总结
print(df2.describe())
# 翻转数据
print(df2.T)
# print(np.transpose(df2))等价于上述操作
'''
axis=1表示行
axis=0表示列
默认ascending(升序)为True
ascending=True表示升序,ascending=False表示降序
下面两行分别表示按行升序与按行降序
'''
print(df2.sort_index(axis=1,ascending=True))
print(df2.sort_index(axis=1,ascending=False))
# 表示按列降序与按列升序
print(df2.sort_index(axis=0,ascending=False))
print(df2.sort_index(axis=0,ascending=True))
# 对特定列数值排列
# 表示对C列降序排列
print(df2.sort_values(by='C', ascending=False))
dates = pd.date_range('20180819', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
print(df)
print(df['A'])
print(df.A)
# 选择跨越多行或多列
# 选取前3行
print(df[0:3])
print(df['2018-08-19':'2018-08-21'])
# 根据标签选择数据
# 获取特定行或列
# 指定行数据
print(df.loc['20180819'])
# 指定列
# 两种方式
print(df.loc[:,'A':'B'])
print(df.loc[:,['A','B']])
# 行与列同时检索
print(df.loc['20180819',['A','B']])
# 根据序列iloc
# 获取特定位置的值
print(df.iloc[3,1])
print(df.iloc[3:5,1:3]) # 不包含末尾5或3，同列表切片
# 跨行操作
print(df.iloc[[1,3,5],1:3])
# 混合选择
# print(df.ix[:3,['A','C']])
print(df.iloc[:3,[0,2]]) # 结果同上
# 通过判断的筛选
print(df[df.A>8])
# 通过判断的筛选
print(df.loc[df.A>8])
print(df.loc['20180819','A':'B'])
print(df.iloc[0,0:2])
# print(df.ix[0,'A':'B'])
# 4.Pandas设置值¶
# 4.1 创建数据
# 创建数据
dates = pd.date_range('20180820',periods=6)
df = pd.DataFrame(np.arange(24).reshape(6,4), index=dates, columns=['A','B','C','D'])
print(df)
# 4.2 根据位置设置loc和iloc
# 根据位置设置loc和iloc
df.iloc[2,2] = 111
df.loc['20180820','B'] = 2222
print(df)
# 3 4.3 根据条件设置
# 根据条件设置
# 更改B中的数，而更改的位置取决于4的位置，并设相应位置的数为0
df.B[df.A>4] = 0
print(df)
df.B.loc[df.A>4] = 0
print(df)
# 4.4 按行或列设置
# 按行或列设置
# 列批处理，F列全改为NaN
df['F'] = np.nan
print(df)
# 4.5 添加Series序列(长度必须对齐)
df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20180820',periods=6))
print(df)
# 4.6 设定某行某列为特定值
# 设定某行某列为特定值
# df.ix['20180820','A'] = 56
# print(df)
# ix 以后要剥离了，尽量不要用了
df.loc['20180820','A'] = 67
print(df)
df.iloc[0,0] = 76
print(df)
# 4.7 修改一整行数据
# 修改一整行数据
df.iloc[1] = np.nan # df.iloc[1,:]=np.nan
print(df)
df.loc['20180820'] = np.nan # df.loc['20180820,:']=np.nan
print(df)
# 5.Pandas处理丢失数据
# 5.1 创建含NaN的矩阵
# 创建含NaN的矩阵
# 如何填充和删除NaN数据?
dates = pd.date_range('20180820',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
print(df)
# a.reshape(6,4)等价于a.reshape((6,4))
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df)
#5.2 删除掉有NaN的行或列
# 删除掉有NaN的行或列
print(df.dropna()) # 默认是删除掉含有NaN的行
print(df.dropna(
    axis=0, # 0对行进行操作;1对列进行操作
    how='any' # 'any':只要存在NaN就drop掉；'all':必须全部是NaN才drop
))
# 删除掉所有含有NaN的列
print(df.dropna(
    axis=1,
    how='any'
))
# 5.3 替换NaN值为0或者其他
# 替换NaN值为0或者其他
print(df.fillna(value=0))
# 5.4 是否有缺失数据NaN
# 是否有缺失数据NaN
# 是否为空
print(df.isnull())
# 是否为NaN
print(df.isna())
# 检测某列是否有缺失数据NaN
print(df.isnull().any())
# 检测数据中是否存在NaN,如果存在就返回True
print(np.any(df.isnull())==True)
# 6.Pandas导入导出
# 6.1 导入数据
# 读取csv
data = pd.read_csv('student.csv')
# 打印出data
print(data)
# 前三行
print(data.head(3))
# 前三行
print(data.head(3))
# 6.2 导出数据
# 将资料存取成pickle
data.to_pickle('student.pickle')

