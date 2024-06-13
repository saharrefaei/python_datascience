import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import jaccard_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, log_loss

df = pd.read_csv(r"C:\Users\sahar\Downloads\heart.csv")
df.replace(['', ' ', 'NA', 'N/A', 'n/a', 'na'], np.nan, inplace=True)
df = df.dropna()

class_distribution = df['output'].value_counts()

print(class_distribution)


X = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']].values
y = df['output'].values

normalize = preprocessing.StandardScaler().fit(X)
X = normalize.transform(X.astype(float))

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=4)

logistic_regression = LogisticRegression(C=0.01, solver='liblinear')
logistic_regression.fit(train_x, train_y)

predict_tree = logistic_regression.predict(test_x)
predict_tree_prob = logistic_regression.predict_proba(test_x)

accuracy_score = metrics.accuracy_score(test_y, predict_tree)
jaccard_score_value = jaccard_score(test_y, predict_tree)
log_loss_value = log_loss(test_y, predict_tree_prob)

print("metrics.accuracy_score:", accuracy_score)
print("metrics.jaccard_score:", jaccard_score_value)
print("metrics.log_loss:", log_loss_value)

conf_matrix = confusion_matrix(test_y, predict_tree)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

