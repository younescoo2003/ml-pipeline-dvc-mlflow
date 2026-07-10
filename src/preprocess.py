import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(X_train, X_test):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use in API
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled

def get_scaler():
    """
    Load saved scaler
    """
    return joblib.load('models/scaler.pkl')