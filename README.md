# Bank Loan Prediction using Random Forest

This project uses Random Forest Classification to predict whether a customer will be approved for a personal loan based on various customer attributes.

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