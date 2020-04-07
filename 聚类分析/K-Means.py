import Pandas学习 as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


beer = pd.read_csv('data.txt', encoding="utf-8", sep=' ')  # 这个beer为自定变量，data.txt为预打开文件要和代码同文件，该函数也可打开.csv
print(beer)  # 打印检测
X = beer[["calories", "sodium", "alcohol", "cost"]]  # 定义表头,数据分类

# 进行聚类分析
# 实例化对象
km = KMeans(n_clusters=3).fit(X)  # n_cluster为K值自己定义
km2 = KMeans(n_clusters=2).fit(X)
print(km.labels_)  # 看一下匹配数据属于得类别

# 看下聚类分析后得结果
beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_
print(beer.sort_values('cluster'))  # 按照cluster进行排序

# 基于每个堆(cluster 1或2)的聚类数据求每个堆的平均值
print(beer.groupby("cluster").mean())
print(beer.groupby("cluster2").mean())

# 取到中心值
centers = beer.groupby("cluster").mean().reset_index()
print(centers)

# 画图--指定字体大小
plt.rcParams['font.size'] = 14
# 画图--指定图中数据颜色
colors = np.array(['red', 'green', 'blue', 'yellow'])
# 画图--输入格式
plt.scatter(beer["calories"], beer["alcohol"],c=colors[beer["cluster"]])
plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')
plt.xlabel("Calories")
plt.ylabel("Alcohol")
plt.show()
# 画散点图用到scatter_matrix 直接从pandas库里调用
pd.plotting.scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster"]], figsize=(10,10))
plt.suptitle("With 3 centroids initialized")
plt.show()
# 上面是3个簇的
# 下面是2个簇的
pd.plotting.scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster2"]], figsize=(10,10))
plt.suptitle("With 2 centroids initialized")
plt.show()
# （标准化）使用StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
# 得到新的结果
km = KMeans(n_clusters=3).fit(X_scaled)
beer["scaled_cluster"] = km.labels_
print(beer.sort_values("scaled_cluster"))
print(beer.groupby("scaled_cluster").mean())
# 更新图像
pd.plotting.scatter_matrix(X, c=colors[beer.scaled_cluster], alpha=1, figsize=(10, 10), s=100)
plt.show()
# 进行评估--轮廓系数
score_scaled = metrics.silhouette_score(X, beer.scaled_cluster)
score = metrics.silhouette_score(X, beer.cluster)
print(score_scaled, score)
# 得到评估值,遍历K值，看哪个K值合适
scores = []
for k in range(2, 20):  # 遍历-打印
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)
print(scores)
# 画一个评估曲线--直观的找到合适的K值
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")
plt.plot(list(range(2, 20)), scores)
plt.show()