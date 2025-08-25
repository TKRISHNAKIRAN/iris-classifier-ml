import streamlit as st
import joblib

# Load the trained model
model, target_names = joblib.load("iris_model.joblib")

# Title
st.title("🌸 Iris Flower Prediction App")

st.write("Enter flower measurements below and predict the species")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Prediction button
if st.button("Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
    flower_name = target_names[prediction]
    st.success(f"✅ The flower is **{flower_name}**")
