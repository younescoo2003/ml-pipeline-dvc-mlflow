import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw_data, get_train_test_split
from src.preprocess import preprocess_data
from src.predict import predict

def test_data_loader():
    """
    Test data loading
    """
    df = load_raw_data()
    assert df.shape[1] == 14  # 13 features + target
    assert len(df) > 0

def test_train_test_split():
    """
    Test train/test split
    """
    X_train, X_test, y_train, y_test = get_train_test_split()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) + len(y_test) == 178  # Total Wine dataset size

def test_predict():
    """
    Test prediction function
    """
    sample = [13.0, 2.5, 2.3, 20.0, 100.0, 2.8, 3.0, 0.3, 1.8, 5.0, 1.0, 3.0, 800.0]
    result = predict(sample)
    assert 'class' in result
    assert 'class_name' in result
    assert 'probabilities' in result
    assert len(result['probabilities']) == 3  # 3 classes

if __name__ == "__main__":
    pytest.main([__file__, "-v"])