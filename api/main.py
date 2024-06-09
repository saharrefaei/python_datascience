import matplotlib.pylab as plt
import pandas as pd
import pylab as pl
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv(r"C:\Users\sahar\Downloads\FuelConsumptionCo2.csv")

df.dropna()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


random =  np.random.rand(len(df)) < 0.8

train = df[random]
test=df[~random]

train_x = np.asanyarray(train[['ENGINESIZE']]) 
train_y = np.asanyarray(train[['CO2EMISSIONS']]) 
test_x = np.asanyarray(test[['ENGINESIZE']]) 
test_y = np.asanyarray(test[['CO2EMISSIONS']]) 


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)


clf = linear_model.LinearRegression()
clf.fit(train_x_poly,train_y)


print("coef :",clf.coef_)
print("intercept : " , clf.intercept_)



XX = np.arange(0.0,10,0.1)

YY = clf.intercept_[0] + clf.coef_[0][1] * XX +  clf.coef_[0][2] * np.power(XX,2)





test_x_poly= poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)
print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )