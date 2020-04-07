# 对于不规整数据集
from sklearn.cluster import DBSCAN
import Pandas学习 as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 读取数据
beer = pd.read_csv('data.txt', encoding="utf-8", sep=' ')
X = beer[["calories", "sodium", "alcohol", "cost"]]
db = DBSCAN(eps=10, min_samples=2).fit(X)

labels = db.labels_
print(db.labels_)
beer['cluster_db'] = labels
print(beer.sort_values('cluster_db'))
print(beer.groupby('cluster_db').mean())

# 绘制图像
colors = np.array(['red', 'green', 'blue', 'yellow'])
pd.plotting.scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10,10), s=100)
plt.show()
# 进行评估聚类评估：轮廓系数
# 得到评估值,遍历K值，看哪个K值合适
scores = []
for k in range(2, 20):  # 遍历-打印
    labels = DBSCAN(min_samples=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)
print(scores)
# 画一个评估曲线--直观的找到合适的K值
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")
plt.plot(list(range(2, 20)), scores)
plt.show()

