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
    df_input['TotalCharges'].fillna(df_input['TotalCharges'].mean(), inplace=True) # Fill with mean of input

    # 2. Xử lý các giá trị "No..."
    df_input.replace('No internet service', 'No', inplace=True)
    df_input.replace('No phone service', 'No', inplace=True)

    # 3. Chuyển đổi gender (nếu form gửi dạng chuỗi)
    df_input['gender'].replace({'Male': 0, 'Female': 1}, inplace=True)

    # 4. Chuyển đổi Yes/No (nếu form gửi dạng chuỗi)
    object_yesno_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                            'PaperlessBilling']
    for col in object_yesno_columns:
        df_input[col].replace({'Yes': 1, 'No': 0}, inplace=True)
        # Xử lý missing values (nếu có)
        df_input[col] = df_input[col].apply(lambda x: 1 if x == 1 else (0 if x == 0 else 0))

    # 5. One-Hot Encoding (di chuyển xuống dưới)
    return df_input


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Lấy *tất cả* dữ liệu từ form
        user_input = request.form.to_dict()
        df_input = pd.DataFrame([user_input])

        # 2. Tiền xử lý *trước* khi one-hot encoding
        df_input = preprocess_input(df_input.copy())

        # 3. One-Hot Encoding *ở đây*
        df_input = pd.get_dummies(data=df_input, columns=['InternetService', 'Contract', 'PaymentMethod'])


        # 4. Xử lý cột thiếu/thừa (sau one-hot encoding)
        missing_cols = set(model_columns) - set(df_input.columns)
        for c in missing_cols:
            df_input[c] = 0
        extra_cols = set(df_input.columns) - set(model_columns)  # Xử lý cột thừa
        if extra_cols:
           print(f"Warning: Extra columns in input: {extra_cols}") # Log cột thừa
        df_input = df_input[model_columns] # Chọn đúng cột, đúng thứ tự


        # 5. Chuyển đổi kiểu dữ liệu
        df_input = df_input.astype(float)

        # 6. Scaling
        df_input[scaler_columns] = scaler.transform(df_input[scaler_columns])

        # 7. Dự đoán
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        probability = probabilities[1] if prediction == 1 else probabilities[0]
        probability_percent = round(probability * 100, 2)

        result = "Khách hàng có khả năng RỜI ĐI!" if prediction == 1 else "Khách hàng sẽ Ở LẠI!"
        return render_template('index.html', prediction_text=f"Có xác suất: {probability_percent}% {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Lỗi: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)