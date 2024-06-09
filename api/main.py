import matplotlib.pylab as plt
import pandas as pd
import pylab as pl
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv(r"C:\Users\sahar\Downloads\FuelConsumptionCo2.csv")

cdf = df[["FUELCONSUMPTION_CITY","CYLINDERS",'FUELCONSUMPTION_COMB','ENGINESIZE','CO2EMISSIONS']]

random = np.random.rand(len(cdf)) < 0.8
print(random)

train = cdf[random]
test = cdf[~random]

train_x =np.asanyarray(train[["FUELCONSUMPTION_CITY","CYLINDERS",'FUELCONSUMPTION_COMB','ENGINESIZE']])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])

regression = linear_model.LinearRegression()
regression.fit(train_x,train_y)
 
print("coef :",regression.coef_)
print("intercept : " , regression.intercept_)

test_y_ = regression.predict(test[["FUELCONSUMPTION_CITY","CYLINDERS",'FUELCONSUMPTION_COMB','ENGINESIZE']])
test_x =np.asanyarray(test[["FUELCONSUMPTION_CITY","CYLINDERS",'FUELCONSUMPTION_COMB','ENGINESIZE']])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])

print("r2_score : " , r2_score(test_y_ ,test_y))


