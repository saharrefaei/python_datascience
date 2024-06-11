import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv(r"C:\Users\sahar\Downloads\insurance_data.csv")

df["sex"] = df["sex"].replace({'female': 1, 'male': 0})
df["smoker"] = df["smoker"].replace({'yes': 1, 'no': 0})
df["region"] = df["region"].map({'northeast': 1, 'southwest': 0})

df.replace(['', ' ', 'NA', 'N/A', 'n/a', 'na'], np.nan, inplace=True)
df = df.dropna()


msk = df[['age',	'sex',	'bmi'	,'children',	'smoker',	'region','charges' ]]
random = np.random.rand(len(df)) <0.8
train = df[random]
test = df[~random]



train_x = np.asanyarray(train[['age',	'sex',	'bmi'	,'children',	'smoker',	'region']])
train_y = np.asanyarray(train[['charges']])
regression = linear_model.LinearRegression()
regression.fit(train_x, train_y)
print('coef:', regression.coef_)
print("intercept:", regression.intercept_)
test_x = np.asanyarray(test[['age',	'sex',	'bmi'	,'children',	'smoker',	'region']])
test_y = np.asanyarray(test[['charges']])
test_y_pred = regression.predict(test_x)
r2 = r2_score(test_y, test_y_pred)
print("r2score:", r2)

plt.figure(figsize=(10, 6))
plt.scatter(test_y, test_y_pred, color='blue', label='Predicted vs Actual')
plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=2, color='red', label='Ideal fit')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.legend()
plt.show()