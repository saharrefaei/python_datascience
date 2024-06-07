import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'C:\Users\sahar\OneDrive\Desktop\ML\Housing.csv')

df['date'] = pd.to_datetime(df['date']).astype('int64') // 10**9  # Converting to int64

dataSe=pd.DataFrame(df)

dataSetNa = dataSe.dropna()

model_1 = LinearRegression()
model_2 = LinearRegression()


model_1.fit(dataSetNa[['price']], dataSetNa['date'])
model_2.fit(dataSetNa[['date']], dataSetNa['price'])

predict_date = model_1.predict([[400000]])
predict_price = model_2.predict([[2013]])

print(f"Predicted waiting for price=400000: {predict_date[0]}")
print(f"Predicted duration for date=2013: {predict_price[0]}")
