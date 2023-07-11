import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 


############# #############################
nutrient = pd.read_csv("nutrient.csv",index_col=0)

### Scaling the Data
scaler = MinMaxScaler()
nutrient_scl = scaler.fit_transform(nutrient)
nutrient_scl = pd.DataFrame(nutrient_scl,
                        columns=nutrient.columns,
                        index=nutrient.index)

## Examining the number of clusters
wss = []
nos = [2,3,4,5,6,7,8,9,10]
for i in nos:
    km = KMeans(n_clusters=i, random_state=23)
    km.fit(nutrient_scl)
    wss.append(km.inertia_)

plt.plot(nos, wss)
plt.scatter(nos,wss, color="red")
plt.title("Scree Plot")
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.show()

### Clustering the data
km = KMeans(n_clusters=5)
km.fit(nutrient_scl)
print(km.inertia_)
print(km.labels_)

clust_nutrient = nutrient.copy()
clust_nutrient['Cluster'] = km.labels_
clust_nutrient.sort_values('Cluster')
clust_nutrient.groupby('Cluster').mean()

####################### US Arrests ####################
USArrests = pd.read_csv("USArrests.csv",index_col=0)

### Scaling the Data
scaler = MinMaxScaler()
USArrests_scl = scaler.fit_transform(USArrests)
USArrests_scl = pd.DataFrame(USArrests_scl,
                        columns=USArrests.columns,
                        index=USArrests.index)

## Examining the number of clusters
wss = []
nos = [2,3,4,5,6,7,8,9,10]
for i in nos:
    km = KMeans(n_clusters=i, random_state=23)
    km.fit(USArrests_scl)
    wss.append(km.inertia_)

plt.plot(nos, wss)
plt.scatter(nos,wss, color="red")
plt.title("Scree Plot")
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.show()

### Clustering the data
km = KMeans(n_clusters=4)
km.fit(USArrests_scl)
print(km.inertia_)
print(km.labels_)

clust_USArrests = USArrests.copy()
clust_USArrests['Cluster'] = km.labels_
clust_USArrests.sort_values('Cluster')
clust_USArrests.groupby('Cluster').mean()

