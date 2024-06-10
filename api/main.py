import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# خواندن داده‌ها
df = pd.read_csv(r"C:\Users\sahar\Downloads\1632300362534233.csv")

# تبدیل ستون 'Price' به float
df['Price'] = df['Price'].apply(lambda x: float(x))

# حذف ردیف‌هایی که مقدار ستون 'Area' آن‌ها '###' است
df = df[df['Area'] != '###']

# حذف کاما و تبدیل ستون 'Area' به float
df['Area'] = df['Area'].str.replace(',', '').astype(float)

# حذف مقادیر خالی از ستون 'Address'
df['Address'].dropna()

# تبدیل مقادیر True/False به 1/0
df["Parking"] = df["Parking"].replace({True: 1, False: 0})
df["Warehouse"] = df["Warehouse"].replace({True: 1, False: 0})
df["Elevator"] = df["Elevator"].replace({True: 1, False: 0})

# اعمال One-Hot Encoding بر روی ستون "Address"
encoded_address = pd.get_dummies(df['Address'], prefix='Address')

# اضافه کردن ستون‌های جدید به دیتافریم
df = pd.concat([df, encoded_address], axis=1)

# حذف ستون "Address" اصلی
df.drop(columns=['Address'], inplace=True)

# انتخاب ویژگی‌ها و قیمت به عنوان ورودی و خروجی مدل
features = ["Area", "Room", "Parking", "Warehouse", "Elevator"] + list(encoded_address.columns)
msk = df[features + ['Price']]


# تقسیم داده‌ها به مجموعه آموزشی و آزمایشی
random = np.random.rand(len(msk)) < 0.8
train = msk[random]
test = msk[~random]

# نرمال‌سازی ستون 'Area'
x_data = df['Area'].values
Area_normalized = x_data / max(x_data)
df['Area'] = Area_normalized

# آماده‌سازی داده‌های آموزشی و آزمایشی
train_x = np.asanyarray(train[features])
train_y = np.asanyarray(train[['Price']])
test_x = np.asanyarray(test[features])
test_y = np.asanyarray(test[['Price']])

# ایجاد و آموزش مدل خطی
regression = linear_model.LinearRegression()
regression.fit(train_x, train_y)

# نمایش ضرایب و عرض از مبدا مدل
print('coef:', regression.coef_)
print("intercept:", regression.intercept_)

# پیش‌بینی بر روی داده‌های آزمایشی
test_y_pred = regression.predict(test_x)

# محاسبه r2_score
r2 = r2_score(test_y, test_y_pred)
print("r2score:", r2)
