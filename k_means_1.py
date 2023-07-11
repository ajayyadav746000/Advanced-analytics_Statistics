import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

data = np.array([[100, 78, 0.3],
                 [90, 87, 1.4],
                 [200, 90, 0.4],
                 [190, 100, 1.9]])

scaler = MinMaxScaler()
scaler.fit_transform(data)

############# #############################
milk = pd.read_csv("milk.csv",index_col=0)

### Scaling the Data
scaler = MinMaxScaler()
milk_scl = scaler.fit_transform(milk)
milk_scl = pd.DataFrame(milk_scl,
                        columns=milk.columns,
                        index=milk.index)

## Examining the number of clusters
wss = []
nos = [2,3,4,5,6,7,8,9,10]
for i in nos:
    km = KMeans(n_clusters=i, random_state=23)
    km.fit(milk_scl)
    wss.append(km.inertia_)

plt.plot(nos, wss)
plt.scatter(nos,wss, color="red")
plt.title("Scree Plot")
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.show()

### Clustering the data
km = KMeans(n_clusters=4, random_state=23)
km.fit(milk_scl)
print(km.inertia_)
print(km.labels_)

clust_milk = milk.copy()
clust_milk['Cluster'] = km.labels_
clust_milk.sort_values('Cluster')
clust_milk.groupby('Cluster').mean()


