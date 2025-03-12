from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import logging
import os

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load model, scaler
try:
    model = pickle.load(open('model_log.pkl', 'rb'))
    model_rf = pickle.load(open('model_rf.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    scaler_columns = pickle.load(open('scaler_columns.pkl', 'rb'))
    model_columns = pickle.load(open('model_columns.pkl', 'rb'))

    model_segmentation_behavior = pickle.load(open('model_segmentation_behavior.pkl', 'rb'))
    scaler_segmentation_behavior = pickle.load(open('scaler_segmentation_behavior.pkl', 'rb'))
    label_encoder_behavior = pickle.load(open('label_encoder_behavior.pkl', 'rb'))

    model_segmentation_demographic = pickle.load(open('model_segmentation_demographic.pkl', 'rb'))
    scaler_segmentation_demographic = pickle.load(open('scaler_segmentation_demographic.pkl', 'rb'))

    logging.info("Đã load thành công các model và scaler.")

except FileNotFoundError as e:
    logging.error(f"Không tìm thấy file model hoặc scaler: {e}")
    exit()
except Exception as e:
    logging.error(f"Lỗi khi load model hoặc scaler: {e}")
    exit()

def convert_total_charges(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def preprocess_input(df_input):
    try:
        # 1. Xử lý TotalCharges
        df_input['TotalCharges'] = df_input['TotalCharges'].apply(convert_total_charges)
        df_input['TotalCharges'].fillna(df_input['TotalCharges'].mean(), inplace=True)

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

    except Exception as e:
        logging.error(f"Lỗi trong preprocess_input: {e}")
        raise

def predict_churn(model, df_input):
    try:
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        probability = probabilities[1] if prediction == 1 else probabilities[0]
        probability_percent = round(probability * 100, 2)
        result_message = "Khách hàng có khả năng RỜI ĐI!" if prediction == 1 else "Khách hàng sẽ Ở LẠI!"
        return f"Có {probability_percent}% khả năng {result_message}"
    except Exception as e:
        logging.error(f"Lỗi trong predict_churn: {e}")
        raise

def preprocess_segmentation_input(df_input, segmentation_choice):
    try:
        if segmentation_choice == 'behavior':
            # Chuyển đổi kiểu dữ liệu
            if 'Recency' in df_input.columns:
                df_input["Recency"] = df_input["Recency"].astype(float)
            if 'NumWebPurchases' in df_input.columns:
                df_input["NumWebPurchases"] = df_input["NumWebPurchases"].astype(float)
            if 'NumStorePurchases' in df_input.columns:
                df_input["NumStorePurchases"] = df_input["NumStorePurchases"].astype(float)
            if 'NumCatalogPurchases' in df_input.columns:
                df_input["NumCatalogPurchases"] = df_input["NumCatalogPurchases"].astype(float)
            if 'MntWines' in df_input.columns:
                df_input["MntWines"] = df_input["MntWines"].astype(float)
            if 'MntMeatProducts' in df_input.columns:
                df_input["MntMeatProducts"] = df_input["MntMeatProducts"].astype(float)
            if 'MntFishProducts' in df_input.columns:
                df_input["MntFishProducts"] = df_input["MntFishProducts"].astype(float)
            if 'MntSweetProducts' in df_input.columns:
                df_input["MntSweetProducts"] = df_input["MntSweetProducts"].astype(float)
            if 'MntGoldProds' in df_input.columns:
                df_input["MntGoldProds"] = df_input["MntGoldProds"].astype(float)
            # Tạo trường Frequency và Monetary (tùy thuộc vào mô hình của bạn)
            df_input["Frequency"] = df_input["NumWebPurchases"] + df_input["NumStorePurchases"] + df_input["NumCatalogPurchases"]
            df_input["Monetary"] = df_input["MntWines"] + df_input["MntMeatProducts"] + df_input["MntFishProducts"] + df_input["MntSweetProducts"] + df_input["MntGoldProds"]
            # Chọn các features cần thiết
            features_segmentation_behavior = ['Recency', 'Frequency', 'Monetary']
            df_input = df_input[features_segmentation_behavior]

            # Scale dữ liệu
            df_scaled = scaler_segmentation_behavior.transform(df_input)
            df_scaled = pd.DataFrame(df_scaled, columns=features_segmentation_behavior, index=df_input.index)
            return df_scaled
        elif segmentation_choice == 'demographic':
            # Chuyển đổi kiểu dữ liệu
            if 'Customer_Age' in df_input.columns:
                df_input["Customer_Age"] = df_input["Customer_Age"].astype(float)
            if 'Income' in df_input.columns:
                df_input["Income"] = df_input["Income"].astype(float)
            
            df_input = pd.get_dummies(df_input, columns=["Education", "Marital_Status"])

            # Danh sách cột chính xác từ lúc huấn luyện
            expected_columns = ["Customer_Age", "Income", "Education_Basic", "Education_Graduation",
                                "Education_Master", "Education_PhD", "Marital_Status_Separated", "Marital_Status_Single"]

            # Thêm các cột bị thiếu và đặt giá trị mặc định là 0
            for col in expected_columns:
                if col not in df_input.columns:
                    df_input[col] = 0

            df_scaled = scaler_segmentation_demographic.transform(df_input[expected_columns])
            df_scaled = pd.DataFrame(df_scaled, columns=expected_columns, index=df_input.index)

            return df_scaled
    except Exception as e:
        logging.error(f"Lỗi trong preprocess_segmentation_input: {e}")
        raise
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-churn', methods=['POST'])
def predict_churn_route():
    try:
        logging.info("Nhận request dự đoán churn...")

        # 1. Lấy dữ liệu từ form
        user_input = request.form.to_dict()
        df_input = pd.DataFrame([user_input])

        # 2. Lấy model được chọn từ form
        selected_model = request.form.get('model_choice', 'logistic')

        # 3. Tiền xử lý dữ liệu dự đoán churn
        df_input = preprocess_input(df_input.copy())

        # 4. One-Hot Encoding cho dự đoán churn
        df_input = pd.get_dummies(data=df_input, columns=['InternetService', 'Contract', 'PaymentMethod'])

        # 5. Xử lý cột thiếu/thừa cho dự đoán churn
        missing_cols = set(model_columns) - set(df_input.columns)
        for c in missing_cols:
            df_input[c] = 0
        df_input = df_input[model_columns]

        # 6. Chuyển đổi kiểu dữ liệu cho dự đoán churn
        df_input = df_input.astype(float)

        # 7. Scaling (chỉ áp dụng cho các cột cần scale) cho dự đoán churn
        df_input[scaler_columns] = scaler.transform(df_input[scaler_columns])

        # 8. Dự đoán churn
        if selected_model == 'logistic':
            prediction_text = predict_churn(model, df_input)
            model_name = "Logistic Regression"
        elif selected_model == 'random_forest':
            prediction_text = predict_churn(model_rf, df_input)
            model_name = "Random Forest"
        else:
            return render_template('index.html', prediction_text="Lỗi: Mô hình không hợp lệ.", selected_model=selected_model)

        logging.info(f"Dự đoán churn thành công: {prediction_text}")

        # 9. Render template
        return render_template(
            'index.html', 
            prediction_text=f"{model_name}: {prediction_text}", 
            segmentation_text=None, selected_model=selected_model, 
            active_form="churn"
        )


    except KeyError as e:
        logging.error(f"Thiếu key trong request: {e}")
        return render_template('index.html', prediction_text=f"Lỗi: Thiếu thông tin {e}.", selected_model=selected_model, segmentation_text=None)
    except ValueError as e:
        logging.error(f"Giá trị không hợp lệ: {e}")
        return render_template('index.html', prediction_text=f"Lỗi: Giá trị không hợp lệ: {e}.", selected_model=selected_model, segmentation_text=None)
    except Exception as e:
        logging.exception("Lỗi không xác định trong quá trình dự đoán churn.")
        return render_template('index.html', prediction_text="Đã xảy ra lỗi. Vui lòng kiểm tra lại thông tin nhập vào.", selected_model=selected_model, segmentation_text=None)
@app.route('/predict-segmentation', methods=['POST'])
def predict_segmentation_route():
    try:
        logging.info("Nhận request phân khúc khách hàng...")

        # 1. Lấy dữ liệu từ form
        user_input = request.form.to_dict()
        # Lấy tiêu chí phân khúc được chọn từ form
        segmentation_choice = request.form.get('segmentation_choice', 'behavior')
        # Xác định các trường cần kiểm tra theo loại phân khúc
        required_fields = {
            'behavior': ['Recency', 'NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases',
                         'MntWines', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'],
            'demographic': ['Customer_Age', 'Income', 'Education', 'Marital_Status']
        }

        # Kiểm tra xem có thiếu dữ liệu không
        for field in required_fields[segmentation_choice]:
            if field not in user_input or user_input[field].strip() == '':
                return render_template('index.html', segmentation_text=f"Lỗi: Vui lòng nhập giá trị cho {field}.", prediction_text=None, selected_model=None)

        # Chuyển dữ liệu thành DataFrame
        df_input = pd.DataFrame([user_input])

        # Tiền xử lý dữ liệu
        df_scaled = preprocess_segmentation_input(df_input.copy(), segmentation_choice)

        # Phân khúc khách hàng
        if segmentation_choice == 'behavior':
            cluster = model_segmentation_behavior.predict(df_scaled)[0]
            segment_label = label_encoder_behavior.inverse_transform([cluster])[0]
            segment_labels = {0: "Khách hàng mới", 1: "Sắp rời bỏ",2: "Trung bình", 3: "Khách hàng VIP"}
            # In danh sách giá trị gốc và giá trị sau khi encode
            # for i, label in enumerate(label_encoder_behavior.classes_):
            #     print(f"Giá trị ban đầu: {label} -> Giá trị mã hóa: {i}")
            segment_text = f"Khách hàng thuộc nhóm: {segment_label}"
            logging.info(f"Input data for segmentation: {df_scaled}")
            logging.info(f"Predicted cluster: {cluster}, Decoded segment: {segment_label}")

        else:
            cluster = model_segmentation_demographic.predict(df_scaled)[0]
            segment_labels = {0: "Trẻ", 1: "Trung niên", 2: "Cao cấp"}

        segment_text = f"Khách hàng thuộc nhóm: {segment_labels.get(cluster, 'Không xác định')}"
        logging.info(f"Phân khúc khách hàng thành công: {segment_text}")

        return render_template(
            'index.html',
            prediction_text=None,
            segmentation_text=segment_text,
            selected_model=None,
            active_form="segmentation"
        )

    except Exception as e:
        logging.exception("Lỗi trong quá trình phân khúc.")
        return render_template('index.html', segmentation_text="Đã xảy ra lỗi. Vui lòng kiểm tra lại thông tin nhập vào.", prediction_text=None, selected_model=None)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))