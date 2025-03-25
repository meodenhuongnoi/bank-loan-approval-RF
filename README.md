# Bank Loan Prediction using Random Forest

This project uses Random Forest Classification to predict whether a customer will be approved for a personal loan based on various customer attributes.

## Project Structure

```
project/
├── data/
│   └── bankloan.csv
├── src/
│   ├── preprocessing.py    # Data loading and preprocessing
│   ├── random_forest.py    # Model training and evaluation
│   └── main.py            # Main execution script
└── requirements.txt        # Project dependencies
```

## Code Structure

The code is organized into three main modules:

1. **preprocessing.py**
   - `load_data()`: Loads the bank loan dataset
   - `preprocess_data()`: Removes irrelevant features (ID, ZIP Code)
   - `split_data()`: Creates train-test split

2. **random_forest.py**
   - `create_parameter_grid()`: Defines hyperparameter search space
   - `train_model()`: Trains Random Forest using GridSearchCV
   - `evaluate_model()`: Calculates model performance metrics
   - `plot_feature_importance()`: Visualizes feature importance

3. **main.py**
   - Orchestrates the entire modeling process
   - Executes pipeline from data loading to evaluation

## Dataset Description

The dataset (`bankloan.csv`) contains customer information with the following features:

- **ID**: Customer ID
- **Age**: Customer's age
- **Experience**: Years of professional experience
- **Income**: Annual income
- **ZIP Code**: Customer's residence zipcode
- **Family**: Number of family members
- **CCAvg**: Credit Card average spending
- **Education**: Education level
- **Mortgage**: Whether customer has a mortgage
- **Securities Account**: Whether customer has a securities account
- **CD Account**: Whether customer has a CD account
- **Online**: Whether customer uses online banking
- **Credit Card**: Whether customer has a credit card
- **Personal Loan**: Target variable (0 = No loan, 1 = Loan approved)

## Analysis Overview

The analysis includes:

1. Data preprocessing:
   - Removal of irrelevant features (ID, ZIP Code)
   - Train-test split (80-20)

2. Model Optimization:
   - Hyperparameter tuning using GridSearchCV
   - Cross-validation with 5 folds

3. Best Model Parameters:
   - max_depth: 10
   - max_features: sqrt
   - max_samples: 0.9
   - min_samples_leaf: 1
   - min_samples_split: 2
   - n_estimators: 200

## Results

- Best Cross-validation Score: 98.63%
- Test Set Accuracy: 99%

## Feature Importance

The analysis includes a visualization of feature importance, helping identify the most influential factors in loan approval decisions.

## Setup and Usage

1. Clone the repository and navigate to the project directory

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the model:
   ```bash
   python src/main.py
   ```

## Requirements

- Python 3.x
- pandas==2.2.3
- numpy==2.2.4
- scikit-learn==1.6.1
- seaborn==0.13.2
- matplotlib==3.10.1