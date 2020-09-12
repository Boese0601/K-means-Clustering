# -*- coding: utf-8 -*-
"""
K-means
场景：商城希望根据用户的年收入和购物指数来对用户进行聚类
"""

# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv("test.csv")
X = dataset.iloc[0:124, 9:13].values
print(X)
# using the elbow method to find the optimal number of clusters
# 创建10个kmeans对象，其集群数分别是1-10，用手肘法则找到最佳的集群数
from sklearn.cluster import KMeans

wcss = []  # 不同组数的组内平方和
for i in range(1, 11):
    # 这里设置集群数为i，使用‘k-means++’初始化中心点（也可以用random）
    kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10,
                    init="k-means++", random_state=0)
    kmeans.fit(X)  # 用X去拟合
    wcss.append(kmeans.inertia_)  # inertia:样本到其最近聚类中心的平方距离之和,即组内平方和
#
# 通过图像能够非常直观的看出“手肘”
plt.plot(range(1, 11), wcss)
plt.title = 'The Elbow Method'
plt.xlabel = 'Number of Clusters'
plt.ylabel = 'WCSS'
plt.show()  # 由图可见，手肘部位大概是5,即最好分成5个聚类

# applying the k-means to the mall dataset
# 创建集群数是5的kmeans对象
kmeans = KMeans(n_clusters=4, max_iter=300, n_init=10,
                init="k-means++", random_state=0)
# y_kmeans 表示的是每一个用户属于的集群
y_kmeans = kmeans.fit_predict(X)  # 拟合并预测
# y_pred = kmeans.predict(np.array([[80, 80]]))
print(y_kmeans)
dataframe=pd.DataFrame({'target':y_kmeans})
dataframe.to_csv("interest clustering result.csv",index=False,sep=',')
#
# # visualizing the clusters
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Careful')  # 有钱但买的少，谨慎的顾客
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Standard')  # 收入中等购买中等，标准的顾客
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Target')  # 有钱且买的多，目标顾客
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Careless')  # 收入低但买的多，不理智的顾客
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Sensible')  # 收入低买的少，理智的顾客
# kmeans.cluster_centers_  # 每个中心点的坐标
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='centroids')
# plt.legend()
# plt.show()