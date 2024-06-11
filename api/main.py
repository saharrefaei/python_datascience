import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
df = pd.read_csv(r"C:\Users\sahar\Downloads\teleCust1000t.csv")
df.replace(['', ' ', 'NA', 'N/A', 'n/a', 'na'], np.nan, inplace=True)
df = df.dropna()

x = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
y = df['custcat'].values

scater = preprocessing.StandardScaler().fit(x)
x = scater.transform(x.astype(float))


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=4)

print(x_train.shape , y_train.shape)
print(x_test.shape , y_test.shape)

K=4
neigh = KNeighborsClassifier(n_neighbors = K ).fit(x_train , y_train)


yhat = neigh.predict(x_test)

print("Train set Accuracy: ", metrics.accuracy_score(y_train , neigh.predict(x_train)))
print("test set Accuracy: ",metrics.accuracy_score(y_test , yhat))


# or :
  
# ks = 10
# mean_acc = np.zeros((ks-1))
# std_acc = np.zeros((ks-1))

# for n in range(1,ks) :
#   neigh = KNeighborsClassifier(n_neighbors = n ).fit(x_train , y_train)
#   yhat = neigh.predict(x_test)
#   mean_acc[n-1] = metrics.accuracy_score(y_test , yhat)

# print("test set Accuracy: ", mean_acc )