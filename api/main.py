import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
df = pd.read_csv(r"C:\Users\sahar\Downloads\heart.csv")
df.replace(['', ' ', 'NA', 'N/A', 'n/a', 'na'], np.nan, inplace=True)
df = df.dropna()
X = df[['age' , 'sex' , 'cp'  ,'trtbps' , 'chol',  'fbs'  ,'restecg' , 'thalachh',  'exng' , 'oldpeak' , 'slp' , 'caa'  ,'thall']].values
y = df['output']



# normalize  = preprocessing.StandardScaler().fit(X)
# X=normalize.transform(X.astype(float))

train_x ,test_x, train_y , test_y = train_test_split(X,y,test_size=0.2 , random_state=4)

DecisionTree = DecisionTreeClassifier(criterion='entropy',max_depth=4)

DecisionTree.fit(train_x,train_y)

predict_tree = DecisionTree.predict(test_x)

print(predict_tree[0:5])
print(test_y[0:5])


accuracy = metrics.accuracy_score(test_y,predict_tree)

print("acciracy : " , accuracy )

conf_matrix = confusion_matrix(test_y, predict_tree)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# محاسبه احتمالات پیش‌بینی مثبت
predict_prob_tree = DecisionTree.predict_proba(test_x)[:, 1]

# رسم منحنی ROC
fpr, tpr, _ = roc_curve(test_y, predict_prob_tree)
roc_auc = roc_auc_score(test_y, predict_prob_tree)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()