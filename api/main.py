import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
df = pd.read_csv(r"C:\Users\sahar\Downloads\drug200.csv")
df.replace(['', ' ', 'NA', 'N/A', 'n/a', 'na'], np.nan, inplace=True)
df = df.dropna()

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df['Drug']



le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_sex.fit(['HIGH','NORMAL','LOW'])
X[:,2] = le_sex.transform(X[:,2]) 


le_sex.fit(['HIGH','NORMAL'])
X[:,3] = le_sex.transform(X[:,3]) 




print(X)

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=4)

print(x_train.shape , y_train.shape)
print(x_test.shape , y_test.shape)

# K=4
desiionTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4).fit(x_train , y_train)


yhat = desiionTree.predict(x_test)

print("test set Accuracy: ",metrics.accuracy_score(y_test , yhat))


