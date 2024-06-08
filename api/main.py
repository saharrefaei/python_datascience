import matplotlib.pylab as plt
import pandas as pd
import pylab as pl
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score
df = pd.read_csv(r"C:\Users\sahar\Downloads\FuelConsumptionCo2.csv")

df.dropna()

random =  np.random.rand(len(df)) < 0.8

train = df[random]
test=df[~random]


regression = linear_model.LinearRegression()

train_x = np.asanyarray(train[['ENGINESIZE' , 'CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regression.fit(train_x,train_y)
print('coef : tetta 1' , regression.coef_)
print('tetta 0 : ' , regression.intercept_)


test_y_ = regression.predict(test[['ENGINESIZE' , 'CYLINDERS','FUELCONSUMPTION_COMB']])
test_x=np.asanyarray(test[['ENGINESIZE' , 'CYLINDERS','FUELCONSUMPTION_COMB']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])

print('r2_score:', r2_score(test_y, test_y_))