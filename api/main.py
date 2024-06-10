import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing  import PolynomialFeatures
df = pd.read_csv(r"C:\Users\sahar\Downloads\1632300362534233.csv")

df.dropna(inplace=True)
df['Price'].apply(lambda x: float(x))


msk = df[[ "Area",  "Room" , "Parking",'Price' , 'Price(USD)']]
plt.scatter(df.Room , df.Price , color = 'green')
plt.show()

random = np.random.rand(len(msk)) < 0.8
train = msk[random]
test = msk[~random]


train_x = np.asanyarray(train[[ "Room" ]])
train_y = np.asanyarray(train[['Price']]) 
test_x = np.asanyarray(test[[ "Room" ]])
test_y = np.asanyarray(test[['Price']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
# ایجاد و آموزش مدل رگرسیون خطی
regression = linear_model.LinearRegression()
regression.fit(train_x_poly, train_y)
print('coef :',regression.coef_)
print("intercept :" , regression.intercept_)

test_x_poly = poly.fit_transform(test_x)
test_y_ = regression.predict(test_x_poly)

print("r2score : " , r2_score(test_y,test_y_))