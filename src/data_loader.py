import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import os

def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.target_names

def save_raw_data():
    df, _ = load_data()
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/wine.csv', index=False)
    print("Raw data saved to data/raw/wine.csv")
    return df

def load_raw_data():
    return pd.read_csv('data/raw/wine.csv')

def get_train_test_split(test_size=0.2, random_state=42):
    df = load_raw_data()
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)