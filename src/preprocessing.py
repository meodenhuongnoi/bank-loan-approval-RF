"""Preprocessing the dataset before training the random forest"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load the bank loan dataset"""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data by removing irrelevant features"""
    # Remove ID and ZIP Code as they are not relevant features
    features_to_drop = ['ID', 'ZIP.Code']
    X = df.drop(features_to_drop + ['Personal.Loan'], axis=1)
    y = df['Personal.Loan']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
