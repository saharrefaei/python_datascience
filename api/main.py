import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pylab, pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances



df = pd.read_csv(r"C:\Users\sahar\Downloads\cars_clus.csv")
 

print(df.shape , 'before')

df[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
       
df=df.dropna()
df = df.reset_index(drop=True)
print(df.shape , 'after')

X = df[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']].values

min_max = MinMaxScaler()
X=MinMaxScaler().fit_transform(X)

distance = euclidean_distances(X,X)


allogm = AgglomerativeClustering(n_clusters=4, linkage='complete')
allogm.fit(distance)
df['cluster']=allogm.labels_


plt.figure(figsize=(10, 7))

colors = ['red', 'green', 'blue', 'purple']
for cluster in np.unique(df['cluster']):
    plt.scatter(df[df['cluster'] == cluster]['engine_s'], df[df['cluster'] == cluster]['horsepow'], 
                label=f'Cluster {cluster}', color=colors[cluster])

plt.xlabel('Engine Size')
plt.ylabel('Horsepower')
plt.title('Clusters of Cars')
plt.legend()
plt.show()