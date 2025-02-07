from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
Pkl_Filename = "rf_tuned.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    model = pickle.load(file)

# Route for homepage
@app.route('/')
def home():
    return render_template('home.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]  # Ensure floats
        final = np.array(features).reshape((1, -1))  # Dynamically reshape
        pred = model.predict(final)[0]

        if pred < 0:
            result = "Error calculating amount!"
        else:
            result = f"Expected amount is {pred:.3f}"

        return render_template('op.html', pred=result)

    except Exception as e:
        return render_template('op.html', pred=f"Error: {str(e)}")

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
