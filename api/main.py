import matplotlib.pylab as plt
import pandas as pd
import pylab as pl
import numpy as np 

df = pd.read_csv(r"C:\Users\sahar\Downloads\FuelConsumptionCo2.csv")



viz = df[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
print(viz)
viz.hist()
plt.show()