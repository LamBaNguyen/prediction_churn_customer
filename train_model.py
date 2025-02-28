import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

# Đọc dữ liệu
df = pd.read_csv('dataset/Telcom-Customer-Churn.csv')

# Xử lý dữ liệu
def convert_total_charges(value):
    try:
        return float(value)
    except ValueError:
        return pd.NA
df['TotalCharges'] = df['TotalCharges'].apply(convert_total_charges)
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Chuyển đổi các giá trị "No internet service" và "No phone service" thành "No"
df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

# Chuyển đổi giới tính về 0-1
df['gender'].replace({'Male': 0, 'Female': 1}, inplace=True)

# Chuyển dữ liệu Yes/No về 0-1
object_yesno_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'PaperlessBilling', 'Churn']
for col in object_yesno_columns:
    df[col].replace({'Yes': 1, 'No': 0}, inplace=True)

# ONE HOT ENCODING cho các cột danh mục
df = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'])

# FEATURE SCALING
scaler = MinMaxScaler()
scaler_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']  # Các cột cần chuẩn hóa
df[scaler_columns] = scaler.fit_transform(df[scaler_columns])

# Tạo tập train-test
X = df.drop(columns=['Churn', 'customerID'])  # Bỏ cột không cần thiết
y = df['Churn']

# Cân bằng dữ liệu với SMOTE
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42, stratify=y_sm)

# Mô hình RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Mô hình Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# # Dự đoán trên tập test
# y_pred_log = log_model.predict(X_test)

# Lưu mô hình, scaler và danh sách cột chuẩn hóa
pickle.dump(log_model, open('model_log.pkl', 'wb'))
pickle.dump(rf_model, open('model_rf.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(scaler_columns, open('scaler_columns.pkl', 'wb'))  # Lưu danh sách cột chuẩn hóa
pickle.dump(X_train.columns.tolist(), open('model_columns.pkl', 'wb'))  # Lưu danh sách cột của mô hình

print("✅ Mô hình đã được lưu vào model.pkl")
