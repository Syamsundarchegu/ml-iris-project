from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    species = ['Setosa', 'Versicolor', 'Virginica']
    return render_template('index.html', prediction_text=f'Predicted Species: {species[prediction[0]]}')

if __name__ == "__main__":
    app.run(debug=True)
