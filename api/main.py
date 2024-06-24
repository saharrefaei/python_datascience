import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# خواندن دیتافریم از فایل CSV
df = pd.read_csv(r"C:\Users\sahar\Downloads\1632560262896716.csv")
print(df.info())

# تبدیل مقادیر "Gender" به 0 و 1
le_sex = LabelEncoder()
df['Gender'] = le_sex.fit_transform(df['Gender'])

# تمیز کردن داده‌ها
print("Shape of dataset before cleaning: ", df.size)
df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
print("Shape of dataset after cleaning: ", df.size)
df.head(5)

# استخراج ویژگی‌ها برای خوشه‌بندی
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# نرمال‌سازی داده‌ها
scaler = StandardScaler()
X = scaler.fit_transform(X)

# اجرای الگوریتم DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(X)

# اضافه کردن برچسب‌های خوشه‌بندی به دیتافریم
df['clustering'] = db.labels_

# رسم پلات خوشه‌بندی
plt.figure(figsize=(10, 6))

# استفاده از دو ویژگی 'Annual Income' و 'Spending Score' برای رسم پلات
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['clustering'], cmap='viridis', marker='o')

# اضافه کردن برچسب‌ها و عنوان
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clustering using DBSCAN')
plt.colorbar()
plt.show()
