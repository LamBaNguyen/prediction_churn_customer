from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import json

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

    # 4. Chuyển đổi Yes/No
    yesno_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                     'PaperlessBilling']
    for col in yesno_columns:
        df_input[col] = df_input[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0, '1': 1, '0': 0}).fillna(0)

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
        df_input = preprocess_input(df_input.copy())

        # 3. One-Hot Encoding
        df_input = pd.get_dummies(data=df_input, columns=['InternetService', 'Contract', 'PaymentMethod'])

        # 4. Xử lý cột thiếu/thừa
        missing_cols = set(model_columns) - set(df_input.columns)
        for c in missing_cols:
            df_input[c] = 0
        df_input = df_input[model_columns]  # Giữ đúng thứ tự cột như trong model_columns

        # 5. Chuyển đổi kiểu dữ liệu
        df_input = df_input.astype(float)

        # 6. Scaling (chỉ áp dụng cho các cột cần scale)
        df_input[scaler_columns] = scaler.transform(df_input[scaler_columns])

        # 7. Dự đoán
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        probability = probabilities[1] if prediction == 1 else probabilities[0]
        probability_percent = round(probability * 100, 2)

        # 8. Tính ảnh hưởng của từng thuộc tính (Feature Importance)
        feature_importance = abs(model.coef_[0])  # Trọng số của Logistic Regression
        feature_impact = {col: round(imp, 4) for col, imp in zip(df_input.columns, feature_importance)}

        # ✅ Chỉ lấy những feature mà khách hàng đã nhập (giá trị khác 0)
        filtered_impact = {feature: impact for feature, impact in feature_impact.items() \
                        #    if df_input[feature].values[0] != 0\
                            }

        # ✅ **Sắp xếp theo mức độ ảnh hưởng giảm dần**
        sorted_impact = sorted(filtered_impact.items(), key=lambda x: x[1], reverse=True)

        # 9. Tạo biểu đồ Pie Chart với Plotly
        labels = [item[0] for item in sorted_impact]
        values = [item[1] for item in sorted_impact]
        fig_bar = px.bar(
            x=values, 
            y=labels,  # Để cột nằm dọc, ta dùng y làm tên thuộc tính
            orientation='h',  # Biểu đồ cột nằm ngang
            # title="Mức độ ảnh hưởng của tất cả các thuộc tính đến quyết định dự đoán",
            labels={'x': "Mức độ ảnh hưởng", 'y': "Yếu tố"},
            text_auto=True
        )
        # fig_bar.update_layout(
        #     width=1000,  # Chiều rộng
        #     height=600,  # Chiều cao
        #     font=dict(size=14)  # Cỡ chữ lớn hơn
        # )
        graph_json = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

        # 10. Hiển thị kết quả
        result_message = "Khách hàng có khả năng RỜI ĐI!" if prediction == 1 else "Khách hàng sẽ Ở LẠI!"
        return render_template(
            'index.html',
            prediction_text=f"Có {probability_percent}% khả năng {result_message}",
            feature_impact=sorted_impact,
            graph_json=graph_json  # Truyền biểu đồ vào HTML
        )

    except Exception as e:
        print(f"Error during prediction: {e}")  # Ghi log lỗi
        return render_template('index.html', prediction_text="Đã xảy ra lỗi. Vui lòng kiểm tra lại thông tin nhập vào.")

if __name__ == '__main__':
    app.run(debug=True)
