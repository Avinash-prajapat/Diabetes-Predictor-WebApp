from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("diabetes_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # DEBUG: print full form data
        print("✅ Form Received:", request.form)

        # Convert all fields to float
        data = [float(request.form.get(field)) for field in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]

        # DEBUG: print processed input
        print("🧠 Model Input:", data)

        # Prediction
        prediction = model.predict(np.array(data).reshape(1, -1))[0]
        result = "✅ Not Diabetic" if prediction == 0 else "⚠️ Diabetic"

        return render_template("index.html", result=result)
    except Exception as e:
        print("❌ Error:", e)
        return render_template("index.html", result=f"❌ Invalid input: {e}")

if __name__ == '__main__':
    app.run(debug=True)
