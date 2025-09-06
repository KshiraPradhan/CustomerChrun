import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

## load the model
model = tf.keras.models.load_model("model.h5")

## load the scaler and encoders
with open("label_encoder_gender.pkl","rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl","rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

## Streamlit App
st.title("Customer Churn Prediction")

## Input fields
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0, value=1000.0)
credit_score = st.number_input('Credit Score', min_value=0, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

## Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One hot encode the geography column
geo_encoded = np.array(onehot_encoder_geo.transform([[geography]]))
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine all input features
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Predict the class
pred = model.predict(input_scaled)
pred_prob = pred[0][0]

st.write(f"Churn probability: {pred_prob:.2f}")

if pred_prob > 0.5:
    st.write(f"Customer will likely to churn.")
else:
    st.write(f"Customer will not likely to churn.")

