import joblib

# Load the trained model + target names
model, target_names = joblib.load("iris_model.joblib")

print("🌸 Iris Flower Prediction App 🌸")

# Ask user to input values
sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

# Prepare input for prediction
features = [[sepal_length, sepal_width, petal_length, petal_width]]

# Predict
prediction = model.predict(features)[0]
flower_name = target_names[prediction]

print(f"✅ Prediction: The flower is **{flower_name}**")
