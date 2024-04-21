from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the dataset
dataset_path = 'Personal Loan.csv'
dataset = pd.read_csv(dataset_path)

# Global variables for storing model and target feature
model = None
target_feature = None


@app.route('/')
def index():
    return render_template('index.html', columns=dataset.columns)


@app.route('/train', methods=['POST'])
def train():
    global model, target_feature

    # Get selected target feature from form
    target_feature = request.form.get('target_feature')

    # Create X and y
    X = dataset.drop(columns=[target_feature])
    y = dataset[target_feature]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Compute accuracy score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Compute correlation matrix
    correlation_matrix = dataset.corr()

    # Generate correlation heatmap visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.tight_layout()

    # Save the visualization to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image to base64
    correlation_visualization = base64.b64encode(buf.read()).decode('utf-8')

    return render_template('train.html', accuracy=accuracy, correlation_visualization=correlation_visualization)


@app.route('/predict', methods=['GET','POST'])
def predict():
    global model, target_feature

    if request.method == 'POST':
        # Get input values for prediction
        input_data = [request.form.get(col) for col in dataset.columns if col != target_feature]

        # Check if number of features matches
        if len(input_data) != len(dataset.columns) - 1:
            error_message = f"Error: Number of input features ({len(input_data)}) doesn't match the model's expectations ({len(dataset.columns) - 1})."
            return error_message

        # Predict using the trained model
        prediction = model.predict([input_data])[0]

        # Convert prediction to "Yes" or "No"
        prediction_result = "Yes" if prediction == 1 else "No"

        # Pass necessary variables to the predict.html template
        return render_template('predict.html', columns=dataset.columns, target_feature=target_feature,
                               prediction=prediction_result)

    # Pass columns and target_feature to the predict.html template
    return render_template('predict.html', columns=dataset.columns, target_feature=target_feature)


if __name__ == '__main__':
    app.run(debug=True)
