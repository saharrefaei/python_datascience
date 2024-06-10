import matplotlib.pylab as plt
import pandas as pd
import pylab as pl
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

# خواندن داده‌ها از فایل CSV
df = pd.read_csv(r"C:\Users\sahar\Downloads\china_gdp.csv")

# رسم نمودار اولیه
plt.figure(figsize=(8,5))
x_data = df['Year'].values 
y_data = df['Value'].values
plt.plot(x_data , y_data , 'ro')
# تعریف تابع سیگموید (لجستیک)
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y

# نرمال‌سازی داده‌ها
xdata = x_data / max(x_data)
ydata = y_data / max(y_data)

# برازش منحنی با استفاده از تابع لجستیک
popt, pcov = curve_fit(sigmoid, xdata, ydata)

# چاپ پارامترهای بهینه
print(popt[0], popt[1])

# رسم منحنی برازش داده شده
x = np.linspace(1960, 2015, 55)
x = x / max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )