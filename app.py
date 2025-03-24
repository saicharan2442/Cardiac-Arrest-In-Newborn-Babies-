from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from cardiac_rf import RandomForest

app = Flask(__name__)

# Load Dataset
DATASET_PATH = "data.csv"
data = pd.read_csv(DATASET_PATH)

# Train Model
X = data.drop(columns=['target']).values
y = data['target'].values

# Train RandomForest Model
rf_model = RandomForest(n_trees=10, max_depth=10, sample_size=0.8)
rf_model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    accuracy = None

    if request.method == 'POST':
        try:
            # Collect all form inputs
            form_data = [float(request.form.get(key, 0)) for key in [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]]

            # Verify input count
            if len(form_data) == 13:
                # Model Prediction
                prediction = rf_model.predict(np.array([form_data]))[0]

                # Calculate Accuracy
                y_pred = rf_model.predict(X)
                accuracy =  round(np.mean(y_pred == y) * 100, 2) 

                result = "Positive" if prediction == 1 else "Negative"
            else:
                result = "Invalid input count. Ensure all 13 fields are filled."
                accuracy = "N/A"
        
        except Exception as e:
            result = f"Error: {str(e)}"
            accuracy = "N/A"

    return render_template('index.html', result=result, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
