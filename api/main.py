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

df = pd.read_csv(r"C:\Users\sahar\Downloads\1632560262896716.csv")
print(df.info())

print(df['Gender'].values)


le_sex = LabelEncoder()
df['Gender'] = le_sex.fit_transform(df['Gender'])

X = df[['Gender','Age',	'Annual Income (k$)','Spending Score (1-100)']].values



Normalize = StandardScaler()

X=Normalize.fit_transform(X)


clusterNumber = 3

k_means = KMeans(init = "k-means++", n_clusters = clusterNumber, n_init = 12)
k_means.fit(X)
df['clustering_label'] = k_means.labels_

# رسم پلات خوشه‌بندی
plt.figure(figsize=(10, 6))

# استفاده از دو ویژگی 'Annual Income' و 'Spending Score' برای رسم پلات
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['clustering_label'], cmap='viridis', marker='o')

# اضافه کردن برچسب‌ها و عنوان
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clustering using KMeans')
plt.colorbar()
plt.show()