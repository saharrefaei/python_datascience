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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv(r"C:\Users\sahar\Downloads\heart.csv")
print(df.shape)

print(df.info())


df.describe()

print ("Shape of dataset before cleaning: ", df.size)
df[[ 'age',	'sex'	,'cp'	,'trtbps',	'chol',	'fbs',	'restecg',	'thalachh'	,'exng',	'oldpeak'	,'slp'	,'caa',	'thall'
]] = df[[ 'age',	'sex'	,'cp'	,'trtbps',	'chol',	'fbs',	'restecg',	'thalachh'	,'exng',	'oldpeak'	,'slp'	,'caa',	'thall']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", df.size)
df.head(5)
df = df.dropna()



print(df.isnull().sum(), "null")
df=df[pd.to_numeric(df['oldpeak'],errors='coerce').notnull()]
df['oldpeak']=df['oldpeak'].astype('int') 





plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
# plt.show()

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


df_cleaned = remove_outliers(df, 'chol')
df_cleaned = remove_outliers(df_cleaned, 'trtbps')



df = df_cleaned




corr_matrix = df.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()




X = df[['restecg','fbs','cp', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']].values
y = df['output'].values

sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
sel.fit(X, y)
selected_feat_idx = sel.get_support()
selected_feat = df[['restecg','fbs','cp', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']].columns[selected_feat_idx]

X = df[selected_feat].values



normalize = preprocessing.StandardScaler().fit(X)
X = normalize.transform(X.astype(float))
print(X)
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
plt.ylabel('hamle qalbi')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()