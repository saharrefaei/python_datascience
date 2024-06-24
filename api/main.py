import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pylab, pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.calibration import LabelEncoder
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

df = pd.read_csv(r"C:\Users\sahar\Downloads\1632560262896716.csv")
print(df.info())


le_sex = LabelEncoder()
df['Gender'] = le_sex.fit_transform(df['Gender'])


print ("Shape of dataset before cleaning: ", df.size)
df[['Gender','Age',	'Annual Income (k$)','Spending Score (1-100)']] = df[['Gender','Age','Annual Income (k$)','Spending Score (1-100)']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", df.size)
df.head(5)

X = df[['Gender','Age',	'Annual Income (k$)','Spending Score (1-100)']].values
print(df)

min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(X)



distnce=euclidean_distances(feature_mtx,feature_mtx) 
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(distnce)

df['clustering']=agglom.labels_


plt.figure(figsize=(10, 6))

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['clustering'], cmap='viridis', marker='o')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clustering using AgglomerativeClustering')
plt.colorbar()
plt.show()

