import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('dataset/marketing_campaign.csv', delimiter=';')

# ================= Data Preprocessing =================
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%Y-%m-%d')
df['Customer_Age'] = 2025 - df['Year_Birth']
df = df[df['Customer_Age'] < 100]
df['Income'].fillna(df['Income'].median(), inplace=True)

# Xử lý tình trạng hôn nhân
df['Marital_Status'] = df['Marital_Status'].replace({'Alone': 'Single', 'YOLO': 'Single', 'Single': 'Single',
                                                      'Married': 'Married', 'Together': 'Married',
                                                      'Divorced': 'Separated', 'Widow': 'Separated',
                                                      'Absurd': 'Unknown'})

# Tạo đặc trưng Frequency & Monetary
df['Frequency'] = df['NumWebPurchases'] + df['NumStorePurchases'] + df['NumCatalogPurchases']
df['Monetary'] = df['MntWines'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']

# ================= Train Customer Segmentation Model =================
features_segmentation = ['Recency', 'Frequency', 'Monetary', 'NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases', 'Income']
X_segmentation = df[features_segmentation].astype(float)

# Chuẩn hóa dữ liệu
scaler_segmentation = StandardScaler()
X_segmentation_scaled = scaler_segmentation.fit_transform(X_segmentation)

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_segmentation_scaled)

# Lưu model & scaler
pickle.dump(kmeans, open('model_segmentation.pkl', 'wb'))
pickle.dump(scaler_segmentation, open('scaler_segmentation.pkl', 'wb'))

print("✅ Mô hình phân khúc khách hàng đã được huấn luyện và lưu thành công!")
