
# Task 4: Logistic Regression (Classification)

## Objective
To build a binary classification model using logistic regression.

## Steps Performed
- Loaded Breast Cancer dataset
- Split data into training and testing sets
- Standardized features using StandardScaler
- Trained Logistic Regression model
- Evaluated using confusion matrix, precision, recall, and ROC-AUC
- Plotted ROC curve

## Evaluation Metrics
- Confusion Matrix: Shows correct and incorrect predictions
- Precision: Accuracy of positive predictions
- Recall: Ability to find all positive cases
- ROC-AUC: Measures model performance

## Insights
- Logistic regression effectively classifies binary outcomes
- Standardization improves model performance
- ROC curve shows trade-off between true positive and false positive rates
- High ROC-AUC indicates strong classification ability
- Model performance depends on threshold selection

## Output
- roc_curve.png
Confusion Matrix:
 [[41  2]
 [ 1 70]]
Precision: 0.9722222222222222
Recall: 0.9859154929577465
ROC-AUC: 0.99737962659679