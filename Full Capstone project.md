# Full Capstone project

## Motivation

Valuation techniques have always played a crucial role in financial decision-making. This project aims to identify the major determinants of variations in financial ratios on the American market using both linear and non-linear models. The focus is on the Price to Book multiple, and its relationship with key financial indicators such as ROE (return on equity), EPS (earnings per share), and BETA (historical risk). The analysis spans from 1989 to 2019, utilizing a dataset scraped from Datastream Refinitiv Eikon and cleaned for outliers and missing values.



### Model Description: Ordinary Least Squares (OLS)

#### Input
The model accepts features labeled as ROE, EPS, and BETA.

#### Output
OLS regression results predicting the target variable PB (Price to Book multiple).

#### Model Architecture
The model utilizes the Ordinary Least Squares regression method, emphasizing statistical techniques for linear regression.

#### Performance
- R-squared: 0.290
- Adjusted R-squared: 0.290
- F-statistic: 2632
- Prob (F-statistic): 0.00
- Log-Likelihood: -24076
- Dataset Size: 19,301 observations

#### Limitations
- Assumption of Linearity
- Sensitivity to Outliers
- Assumption of Independence
- Normality of Residuals

#### Trade-offs
- Interpretability
- Assumption Strictness
- Limited to Linear Relationships
- Data Size Sensitivity

### Model Description: Logit Regression

#### Input
Features include ROE, EPS, BETA, and sector information.

#### Output
Binary predictions for PB_binary with regression coefficients, statistics, and evaluation metrics.

#### Model Architecture
Logistic regression model with 13 features.

#### Performance
- Pseudo R-squared: 0.1931
- Log-Likelihood: -11,200
- LL-Null: -13,880
- LLR p-value: 0.000
- Dataset Size: 22,058 observations

#### Limitations
- Assumption of Linearity
- Sensitivity to Outliers
- Multicollinearity
- Assumption of Independence

#### Trade-offs
- Interpretability
- Assumption Strictness
- Simplicity
- Threshold Selection

### Model Description: K-Nearest Neighbors (KNN)

#### Input
The model uses the KNN algorithm with varying numbers of neighbors.

#### Output
Binary predictions with RMSE and Misclassification Rate, highlighting the best-performing configuration.

#### Model Architecture
KNN algorithm with different numbers of neighbors.

#### Performance
- Best Neighbors: 10
- Best RMSE: 0.4851
- Best Misclassification Rate: 0.2354
- Dataset Size: 5,515 observations

#### Limitations
- Sensitivity to Scale
- Computational Intensity
- Choice of K

#### Trade-offs
- Simplicity
- Lack of Model Interpretability
- Dependency on Training Data
- Impact of Outliers

### Model Description: Random Forest

#### Input
The model uses the Random Forest algorithm for binary classification.

#### Output
Binary predictions with evaluation metrics.

#### Model Architecture
Random Forest algorithm.

#### Performance
- Accuracy: 0.7628
- Precision: 0.6631
- Recall: 0.5280
- Dataset Size: 5,515 observations

#### Limitations
- Lack of Interpretability
- Computational Intensity
- Overfitting

#### Trade-offs
- High Predictive Power
- Ensemble of Trees
- Feature Importance
- Model Complexity

### Model Description: Random Forest Classifier with Hyperparameter Tuning

#### Input
The model utilizes a Random Forest classifier with hyperparameter tuning.

#### Output
Binary predictions with optimized hyperparameters.

#### Model Hyperparameters
- max_depth: 12
- max_features: 0.1
- min_samples_leaf: 1
- min_samples_split: 6
- n_estimators: 150

#### Performance
- Best Accuracy: 0.7628
- Dataset Size: 5,515 observations

#### Limitations
- Computational Intensity

#### Trade-offs
- Accuracy vs. Computational Cost

