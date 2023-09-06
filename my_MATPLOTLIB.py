import pandas as pd
import numpy as np  # Add this import statement
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from flask import Flask, request, jsonify

# Read the Excel file
file_path = '/Users/saaduddinbaig/Downloads/customer_churn_large_dataset.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Check the structure of the DataFrame
print(df.head())  # Display the first few rows of the DataFrame
print(df.columns)  # Print the column names

# Check if 'categorical_feature' is present in the DataFrame
if 'categorical_feature' not in df.columns:
    print("'categorical_feature' column not found in the DataFrame. Please check the column name.")
else:
    # Handle missing data
    df.dropna(inplace=True)

    # Encode categorical variables
    encoder = OneHotEncoder(drop='first', sparse=False)
    
    # Check if 'categorical_feature' is in the DataFrame after removing missing values
    if 'categorical_feature' in df.columns:
        encoded_features = encoder.fit_transform(df[['categorical_feature']])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names(['categorical_feature']))

        # Combine encoded features with numerical features
        X = pd.concat([encoded_df, df[['numerical_feature1', 'numerical_feature2']]], axis=1)

        # Target variable
        y = df['churn']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize models
        logistic_regression = LogisticRegression()
        random_forest = RandomForestClassifier()

        # Train models
        logistic_regression.fit(X_train_scaled, y_train)
        random_forest.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_lr = logistic_regression.predict(X_test_scaled)
        y_pred_rf = random_forest.predict(X_test_scaled)

        # Evaluate models
        def evaluate_model(y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            return accuracy, precision, recall, f1

        accuracy_lr, precision_lr, recall_lr, f1_lr = evaluate_model(y_test, y_pred_lr)
        accuracy_rf, precision_rf, recall_rf, f1_rf = evaluate_model(y_test, y_pred_rf)

        print("Logistic Regression Metrics:")
        print(f"Accuracy: {accuracy_lr}")
        print(f"Precision: {precision_lr}")
        print(f"Recall: {recall_lr}")
        print(f"F1 Score: {f1_lr}")

        print("\nRandom Forest Metrics:")
        print(f"Accuracy: {accuracy_rf}")
        print(f"Precision: {precision_rf}")
        print(f"Recall: {recall_rf}")
        print(f"F1 Score: {f1_rf}")

        # Define hyperparameters to search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            # Add other hyperparameters here
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5)

        # Perform hyperparameter tuning
        grid_search.fit(X_train_scaled, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Evaluate the best model
        y_pred_best = best_model.predict(X_test_scaled)
        accuracy_best, precision_best, recall_best, f1_best = evaluate_model(y_test, y_pred_best)

        print("Best Random Forest Metrics:")
        print(f"Accuracy: {accuracy_best}")
        print(f"Precision: {precision_best}")
        print(f"Recall: {recall_best}")
        print(f"F1 Score: {f1_best}")

        # Create a Flask web application
        app = Flask(__name__)

        # Define a route for making predictions
        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                # Get the input data from the request
                data = request.json
                
                # Preprocess the input data (similar to what was done for training data)
                # For example, if your input data has 'categorical_feature', 'numerical_feature1', and 'numerical_feature2':
                encoded_data = encoder.transform([[data['categorical_feature']]])
                numerical_data = [data['numerical_feature1'], data['numerical_feature2']]
                input_features = np.concatenate((encoded_data, numerical_data), axis=1)
                scaled_features = scaler.transform(input_features)
                
                # Make a prediction using the best model
                prediction = best_model.predict(scaled_features)
                
                # Convert the prediction to a human-readable format (e.g., 0 for not churned, 1 for churned)
                result = {'prediction': int(prediction[0])}
                
                return jsonify(result)
            
            except Exception as e:
                return jsonify({'error': str(e)})

        if __name__ == '__main__':
            app.run(debug=True)
