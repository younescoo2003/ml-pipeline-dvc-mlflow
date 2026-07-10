import mlflow
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import get_train_test_split
from preprocess import preprocess_data

def evaluate_model():
    """Evaluate trained model and generate reports"""
    
    # Load model and scaler
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Load test data
    X_train, X_test, y_train, y_test = get_train_test_split()
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    df_report.to_csv('reports/classification_report.csv')
    print("Classification report saved to reports/classification_report.csv")
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('reports/confusion_matrix.png')
    print("Confusion matrix saved to reports/confusion_matrix.png")
    plt.show()
    
    return df_report

if __name__ == "__main__":
    evaluate_model()