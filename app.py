import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

#set the page
st.set_page_config(
   page_title= "Medical Insurance Cost Prediction",
 )

# CUSTOM CSS
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

/* Title */
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #1f2937;
    text-align: center;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-text {
    text-align: center;
    color: #4b5563;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Form container */
.form-container {
    background-color: white;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}

/* Prediction box */
.prediction-box {
    background-color: #ecfdf5;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #065f46;
    margin-top: 25px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #4facfe, #00f2fe);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #43e97b, #38f9d7);
}

/* Input boxes */
.stNumberInput input {
    border-radius: 10px;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# Load the model
model = joblib.load("linear_regression_model.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

#Load columns
model_columns = joblib.load("model_columns.pkl")

# Title

st.markdown(
    '<div class="main-title">🏥 Medical Insurance Cost Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">Predict insurance cost based on patient details</div>',
    unsafe_allow_html=True
)


#User Inputs

col1,col2 = st.columns(2)
with col1:
 age = st.number_input("Age", min_value = 18, max_value = 100)
 sex = st.selectbox("Sex",["male","female"])
 bmi = st.number_input("BMI", min_value = 10.0, max_value = 60.0)

with col2: 
 children = st.number_input("Number of Children", min_value = 0, max_value= 10)
 smoker = st.selectbox("Smoker",["yes", "no"])
 region = st.selectbox("Region",["northeast","northwest","southeast","southwest"])

#Center Button
col1,col2,col3 = st.columns(3)
with col2:
 predict_button = st.button("Predict Insurance Cost")

 st.markdown('</div>', unsafe_allow_html=True)

#Create Input Dictionary

if predict_button:
  input_dict = {
    'age' : [age],
    'sex' : [sex],
    'bmi' : [bmi],
    'children' : [children],
    'smoker' : [smoker],
    'region' : [region]
    }

  # Convert to Dataframe
  input_df = pd.DataFrame(input_dict)

  #Apply get dummies
  input_df = pd.get_dummies(input_df)


  #Reindex Columns
  input_df = input_df.reindex(columns = model_columns, fill_value = 0)


  #scale data
  input_scaled = scaler.transform(input_df) 

  with st.spinner("Calculating Insurance Cost..."):
    time.sleep(2)

    # Predict
    prediction = model.predict(input_scaled)[0]

  # Remove Negative Prediction
  prediction = max(0, prediction)

  # Toast Notification
  st.toast("Prediction Completed Successfully ✅")

  # Display Prediction
  st.markdown(
        f"""
        <div class="prediction-box">
            Predicted Insurance Cost <br><br>
             {prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )



