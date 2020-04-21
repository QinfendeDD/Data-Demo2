from sklearn import datasets,linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#读取数据
data=datasets.load_diabetes()
X=data['data'][:,np.newaxis,2] # 挑选一个特征：BMI
y=data['target']
#线性回归
lr=linear_model.LinearRegression()
lr.fit(X,y)
y_pred=lr.predict(X)
#画图
sns.set(style='darkgrid')
plt.plot(X,y,'.k')
plt.xlabel('BMI')
plt.ylabel('quantitative measure of disease progression')
plt.plot(X,y_pred,'-r',linewidth=2,label='Ordinary Least Squares')
plt.legend()
plt.show()


