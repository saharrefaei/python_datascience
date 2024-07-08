import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target

nan_counts = df.isnull().sum()

df_cleaned_rows = df.dropna()
df = df_cleaned_rows

X = df.drop(columns=['target'])
Y = df['target']

min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(feature_mtx, Y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

ridge_clf = RidgeClassifierCV()

ridge_clf.fit(X_train, y_train)

ridge_y_pred = ridge_clf.predict(X_test)

ridge_f1 = f1_score(y_test, ridge_y_pred)
print(f'RidgeClassifierCV F1 Score: {ridge_f1}')

ridge_cm = confusion_matrix(y_test, ridge_y_pred)
cmd = ConfusionMatrixDisplay(ridge_cm, display_labels=['malignant', 'benign'])
cmd.plot()
plt.title('Confusion Matrix for RidgeClassifierCV')
plt.show()



