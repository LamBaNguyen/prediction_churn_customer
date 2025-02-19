from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and column names
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
scaler_columns = pickle.load(open('scaler_columns.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))  # Columns *after* one-hot encoding


def convert_total_charges(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def preprocess_input(df_input):
    # 1. Xử lý TotalCharges
    df_input['TotalCharges'] = df_input['TotalCharges'].apply(convert_total_charges)
    df_input['TotalCharges'].fillna(df_input['TotalCharges'].mean(), inplace=True)  # Điền bằng giá trị trung bình *của input*

    # 2. Xử lý các giá trị "No..."
    df_input.replace('No internet service', 'No', inplace=True)
    df_input.replace('No phone service', 'No', inplace=True)

    # 3. Chuyển đổi gender
    df_input['gender'].replace({'Male': 0, 'Female': 1}, inplace=True)

    # 4. Chuyển đổi Yes/No (tự động hóa bằng vòng lặp)
    yesno_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                     'PaperlessBilling']
    for col in yesno_columns:
        # Xử lý trường hợp giá trị không phải là chuỗi 'Yes'/'No' (ví dụ: số 0/1 từ select)
        df_input[col] = df_input[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0, '1': 1, '0': 0}).fillna(0)
        # .map() xử lý cả chuỗi và số, .fillna(0) cho các giá trị khác

    # 5. One-Hot Encoding (được thực hiện *sau* các bước trên)
    return df_input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Lấy dữ liệu từ form và tạo DataFrame
        user_input = request.form.to_dict()
        df_input = pd.DataFrame([user_input])

        # 2. Tiền xử lý dữ liệu *trước* one-hot encoding
        df_input = preprocess_input(df_input.copy())  # Tạo bản sao để tránh thay đổi dữ liệu gốc

        # 3. One-Hot Encoding
        df_input = pd.get_dummies(data=df_input, columns=['InternetService', 'Contract', 'PaymentMethod'])
        # 4. Xử lý cột thiếu/thừa (quan trọng nhất)
        missing_cols = set(model_columns) - set(df_input.columns)
        for c in missing_cols:
            df_input[c] = 0
        df_input = df_input[model_columns]  # Giữ đúng thứ tự cột như trong model_columns

        # 5. Chuyển đổi kiểu dữ liệu (sau one-hot encoding)
        df_input = df_input.astype(float)

        # 6. Scaling (chỉ áp dụng cho các cột đã được scale trong quá trình huấn luyện)
        df_input[scaler_columns] = scaler.transform(df_input[scaler_columns])


        # 7. Dự đoán
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        probability = probabilities[1] if prediction == 1 else probabilities[0]  # Xác suất của lớp dự đoán
        probability_percent = round(probability * 100, 2)

        result_message = "Khách hàng có khả năng RỜI ĐI!" if prediction == 1 else "Khách hàng sẽ Ở LẠI!"
        return render_template('index.html', prediction_text=f"Có {probability_percent}% khả năng {result_message}")


    except Exception as e:
        # Xử lý lỗi tốt hơn (ghi log và hiển thị thông báo thân thiện)
        print(f"Error during prediction: {e}")  # Ghi log lỗi
        return render_template('index.html', prediction_text="Đã xảy ra lỗi. Vui lòng kiểm tra lại thông tin nhập vào.")
        # Hoặc:  return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)