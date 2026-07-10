import joblib
import numpy as np
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model_and_scaler():
    """
    Load trained model and scaler using absolute paths
    """
    model_path = os.path.join(PROJECT_ROOT, 'models', 'model.pkl')
    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict(features):
    """
    Make prediction on new data
    """
    try:
        model, scaler = load_model_and_scaler()
        
        # Convert to array and scale
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        class_names = ['Class 0', 'Class 1', 'Class 2']
        
        return {
            'class': int(prediction[0]),
            'class_name': class_names[prediction[0]],
            'probabilities': probabilities[0].tolist()
        }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise