from preprocessing import load_data, preprocess_data, split_data
from random_forest import train_model, evaluate_model, plot_feature_importance

def main():
    # Load and preprocess data
    df = load_data('data/bankloan.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    print("Best parameters:", results['best_params'])
    print("Best cross-validation score:", results['best_cv_score'])
    print("Test set accuracy:", results['test_accuracy'])
    
    # Plot feature importance
    feature_importance = plot_feature_importance(model, X_train.columns)

if __name__ == "__main__":
    main()
