import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import pandas as pd
import json
from data_loader import get_train_test_split
from preprocess import preprocess_data

def train_model():
    """
    Train Random Forest model with MLflow tracking
    """
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = get_train_test_split()
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    # Model parameters
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    mlflow.set_experiment("random_forest_wine")
    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_wine"):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Save model locally
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/model.pkl')
        
        # Save metrics to JSON (for DVC)
        os.makedirs('reports', exist_ok=True)
        with open('reports/metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        print(f"Model trained and saved!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   MLflow Run ID: {mlflow.active_run().info.run_id}")
        
        return model, params, metrics

if __name__ == "__main__":
    train_model()