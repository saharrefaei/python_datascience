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

random = np.random.rand(len(cdf))<0.8

train = cdf[random]
test = cdf[~random]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)

train_x_poly = poly.fit_transform(train_x)

regression = linear_model.LinearRegression()
regression.fit(train_x_poly , train_y)

print("coef :" , regression.coef_)
print("intercept : " , regression.intercept_)

test_x_poly = poly.fit_transform(test_x)
test_y_ = regression.predict(test_x_poly)

print("r2_score :" , r2_score(test_y, test_y_))