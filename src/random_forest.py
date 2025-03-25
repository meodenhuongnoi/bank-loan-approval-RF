import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def create_parameter_grid():
    """Define parameter grid for GridSearchCV"""
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'max_samples': [0.5, 0.7, 0.9, None]
    }

def train_model(X_train, y_train):
    """Train Random Forest model using GridSearchCV"""
    rf = RandomForestClassifier(criterion='gini', random_state=42)
    param_grid = create_parameter_grid()
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {
        'best_params': model.best_params_,
        'best_cv_score': model.best_score_,
        'test_accuracy': accuracy
    }

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.best_estimator_.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance in Random Forest Model')
    plt.show()
    
    return feature_importance 