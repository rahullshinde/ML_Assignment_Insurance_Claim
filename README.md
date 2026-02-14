
# Car Insurance Claim Prediction

## a. Problem Statement
Insurance companies need to accurately predict whether a customer is likely to file an insurance claim. Incorrect predictions can lead to financial losses or missed risk mitigation opportunities.
The objective of this project is to predict whether a customer will file an insurance claim based on demographic, driving, and vehicle-related features.
This is a binary classification problem where:
- 0 = No claim
- 1 = Claim

## b. Dataset Description
The dataset contains customer and vehicle information related to car insurance policies. Each row represents a single policyholder.

Target Variable:<br>
OUTCOME<br>
0 → No insurance claim<br>
1 → Insurance claim<br>

Key Features:<br>
Demographic details: AGE, GENDER, RACE, EDUCATION, INCOME, MARRIED, CHILDREN<br>
Driving behavior: DRIVING_EXPERIENCE, SPEEDING_VIOLATIONS, DUIS, PAST_ACCIDENTS<br>
Vehicle information: VEHICLE_TYPE, VEHICLE_YEAR, VEHICLE_OWNERSHIP, ANNUAL_MILEAGE<br>
Financial indicator: CREDIT_SCORE<br>

Categorical features were encoded and numerical features were scaled using a preprocessing pipeline.

## c. Models Used
1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Model Comparison Table

| ML Model Name       |   Accuracy |   AUC |   Precision |   Recall |    F1 |   MCC |
|---------------------|------------|-------|-------------|----------|-------|-------|
| Logistic Regression |      0.841 | 0.915 |       0.839 |    0.841 | 0.839 | 0.623 |
| Decision Tree       |      0.842 | 0.902 |       0.843 |    0.842 | 0.842 | 0.634 |
| kNN                 |      0.825 | 0.884 |       0.822 |    0.825 | 0.823 | 0.585 |
| Naive Bayes         |      0.804 | 0.868 |       0.799 |    0.804 | 0.799 | 0.526 |
| Random Forest       |      0.840 | 0.910 |       0.839 |    0.840 | 0.839 | 0.624 |
| XGBoost             |      0.842 | 0.916 |       0.842 |    0.842 | 0.842 | 0.630 |

## Model Observation

| ML Model Name            | Observation about model performance                                                                                                                                                                            |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Provides balanced and stable performance with strong AUC (0.915) and consistent precision and recall (~0.84). Serves as a solid baseline model with good generalization ability.                               |
| Decision Tree            | Shows competitive accuracy (0.842) and good precision, but slightly lower AUC (0.902). May suffer from overfitting compared to ensemble methods, though still performs well overall.                           |
| kNN                      | Moderate performance with AUC (0.884) and MCC (0.585). Performance may be affected by feature scaling and sensitivity to noise, resulting in slightly reduced effectiveness compared to tree-based models.     |
| Naive Bayes              | Lowest overall performance among all models (Accuracy: 0.804, MCC: 0.526). Assumption of feature independence likely limits predictive capability, though it remains computationally efficient & simple.       |
| Random Forest (Ensemble) | Strong performance across all metrics (Accuracy: 0.840, AUC: 0.910). More robust than single Decision Tree and better at handling complex feature interactions.                                                |
| XGBoost (Ensemble)       | Best overall performing model with highest AUC (0.916) & strong balance across precision, recall, and F1 score (≈0.842). Demonstrates superior generalization & ability to capture subtle patterns in dataset. |

