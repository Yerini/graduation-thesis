# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np #useful numbers
import matplotlib.pyplot as plt #for graphing
from matplotlib import style 
style.use("ggplot")
from sklearn.cluster import KMeans #for clustering

x = [1,5,1.5,8,1,9,5,11,9,8,7,2,1,1.3,1.6,5.9,3.7,4.6,11.3,5.6,3.7,9,10]
y = [2,8,1.8,8,0.6,11,3.2,2.8,5.6,4.3,2.5,3.7,2.8,5.2,3.8,1.5,6.8,3.6,4.7,2.5,5.7,8.9,1.2]

plt.scatter(x,y)
plt.show()

xa = np.array([[1,2],[5, 8],
              [1.5, 1.8],
              [8, 8],
              [1, 0.6],
              [9, 11],[5,3.2],[11,2.8],[9,5.6],[8,4.3],[7,2.5],[2,3.7],[1,2.8],[1.3,5.2],[1.6,3.8],[5.9,1.5],
              [3.7,6.8],[4.6,3.6],[11.3,4.7],
              [5.6,2.5],[3.7,5.7],
              [9,8.9],[10,1.2]])

kmeans = KMeans(n_clusters = 3) #cluster 개수 2개로 initialize
kmeans.fit(xa) #fit the data, learning

centroids = kmeans.cluster_centers_ #founding centroids (중심점)
labels = kmeans.labels_
print(centroids)
print(labels)

colors = ["g.","r.","y."]

for i in range(len(xa)):
    print("coordinate:", xa[i], "label:",labels[i])
    plt.plot(xa[i][0],xa[i][1], colors[labels[i]], markersize = 10)
    
plt.scatter(centroids[:,0],centroids[:,1], marker = "x", s = 150,  linewidths = 5, zorder = 10)
plt.show() #centroids x로 표시