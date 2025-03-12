import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

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
df['R_Score'] = pd.qcut(df['Recency'], 4, labels=[4,3,2,1])
df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
df['M_Score'] = pd.qcut(df['Monetary'].rank(method='first'), 4, labels=[1,2,3,4])

# Gộp thành một RFM Score duy nhất
df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
#Gán nhãn
def assign_rfm_segment(score):
    if score in ['444', '443', '434', '344']:
        return 'VIP'
    elif score in ['411', '412', '421']:
        return 'Sắp rời bỏ'
    elif score in ['111', '211', '121']:
        return 'Khách hàng mới'
    else:
        return 'Trung bình'

df['Segment'] = df['RFM_Score'].apply(assign_rfm_segment)

# ================= Train Customer Segmentation Model =================
## 1. Phân khúc theo hành vi mua sắm
# features_behavior = ['Recency', 'Frequency', 'Monetary', 'NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases', 'Income']
# X_behavior = df[features_behavior].astype(float)

# scaler_behavior = StandardScaler()
# X_behavior_scaled = scaler_behavior.fit_transform(X_behavior)

# kmeans_behavior = KMeans(n_clusters=3, random_state=42)
# kmeans_behavior.fit(X_behavior_scaled)

#chosse features
features = ['Recency','Frequency','Monetary']
X = df[features] 
y = df['Segment']

#encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
#scale data
scaler_behavior = StandardScaler()

X_train_scaler = scaler_behavior.fit_transform(X_train)
X_test_scaler = scaler_behavior.transform(X_test)
#xử lý mất câ bằng dữ liệu
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaler, y_train)
#train model 
rf_behavior_model  = RandomForestClassifier(n_estimators=100, random_state=42,class_weight="balanced",max_depth = 5)
rf_behavior_model .fit(X_train_balanced,y_train_balanced)

## 2. Phân khúc theo nhân khẩu học
features_demographic = ["Customer_Age", "Income", "Education", "Marital_Status"]
df_encoded = pd.get_dummies(df, columns=["Education", "Marital_Status"])

# Đảm bảo danh sách cột cố định
expected_columns = ["Customer_Age", "Income", "Education_Basic", "Education_Graduation", "Education_Master", "Education_PhD",
                    "Marital_Status_Separated", "Marital_Status_Single"]
for col in expected_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0  # Điền cột thiếu bằng 0

X_demographic = df_encoded[expected_columns].astype(float)
scaler_demographic = StandardScaler()
X_demographic_scaled = scaler_demographic.fit_transform(X_demographic)

kmeans_demographic = KMeans(n_clusters=3, random_state=42)
kmeans_demographic.fit(X_demographic_scaled)


# Lưu model & scaler
pickle.dump(rf_behavior_model , open('model_segmentation_behavior.pkl', 'wb'))
pickle.dump(scaler_behavior, open('scaler_segmentation_behavior.pkl', 'wb'))
pickle.dump(le, open('label_encoder_behavior.pkl', 'wb'))
pickle.dump(kmeans_demographic, open('model_segmentation_demographic.pkl', 'wb'))
pickle.dump(scaler_demographic, open('scaler_segmentation_demographic.pkl', 'wb'))

print("✅ Mô hình phân khúc khách hàng đã được huấn luyện và lưu thành công!")
