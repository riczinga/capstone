## Model Card

### Model Description OLS

##### Input 
The model accepts input data with features labeled as ROE, EPS, and BETA.

##### Output 
The model performs Ordinary Least Squares (OLS) regression on the selected features (ROE, EPS, BETA) with the target variable PB and provides regression results as text.

##### Model Architecture
The model utilizes the Ordinary Least Squares regression method. It doesn't have a neural network architecture but rather relies on statistical techniques for linear regression.

##### Performance
The performance of the model is summarized by the following regression results:

R-squared: 0.290

Adjusted R-squared: 0.290

F-statistic: 2632.

Prob (F-statistic): 0.00

Log-Likelihood: -24076.

The model was evaluated on a dataset with 19,301 observations, and the performance metrics indicate the goodness of fit for the regression model using features ROE, EPS, and BETA to predict the target variable PB.

##### Limitations
The model has several limitations, including:

Assumption of Linearity: OLS assumes a linear relationship between the target variable PB and the independent variables (ROE, EPS, BETA).

Sensitivity to Outliers: OLS can be sensitive to outliers in the data, impacting the estimated coefficients.

Assumption of Independence: OLS assumes that observations are independent, and violations of this assumption may affect the validity of results.

Normality of Residuals: The model assumes that residuals are normally distributed.

##### Trade-offs
The trade-offs of using OLS regression with features ROE, EPS, and BETA for predicting PB include:

Interpretability: OLS provides coefficients that are easily interpretable in terms of the relationship between ROE, EPS, BETA, and PB.

Assumption Strictness: OLS has assumptions that need to be satisfied for valid results, and violations of these assumptions may impact the reliability of the model.

Limited to Linear Relationships: OLS is effective for linear relationships but may not capture complex non-linear patterns in the data.

Data Size: OLS can handle large datasets, but its performance may be affected by multicollinearity and other issues in high-dimensional data.

Consideration of these trade-offs is crucial when deciding to use OLS regression with the specified features for a particular analysis.

## Model Description Logit regression

**Input:** The model accepts input data with features including ROE, EPS, BETA, and sector information.

**Output:** The model performs Logistic Regression using Maximum Likelihood Estimation (MLE) and predicts the binary variable PB_binary. The output includes regression coefficients, statistics, a confusion matrix, classification report, and evaluation metrics.

**Model Architecture:** The model is a logistic regression model with 13 features (including ROE, EPS, BETA, and sector indicators).

## Performance

The performance of the model is summarized by the following logistic regression results:

- **Convergence:** Optimization terminated successfully.
- **Current function value:** 0.507751
- **Iterations:** 6
- **No. Observations:** 22,058
- **Pseudo R-squared:** 0.1931
- **Log-Likelihood:** -11,200
- **LL-Null (Log-Likelihood Null):** -13,880.
- **LLR p-value (Likelihood Ratio Test):** 0.000

### Coefficients:

- **const:** -1.6221
- **ROE:** 1.4526
- **EPS:** 0.0361
- **BETA:** -0.0571
- **sector_Basic Materials:** 0.6844
- **sector_Consumer Cyclicals:** 0.6741
- **sector_Consumer Non-Cyclicals:** 0.2044
- **sector_Energy:** 0.8421
- **sector_Financials:** 1.0865
- **sector_Healthcare:** 0.6199
- **sector_Industrials:** 0.7969
- **sector_Real Estate:** 1.5362
- **sector_Technology:** 0.7279
- **sector_Utilities:** 1.5652

### Model Evaluation Metrics:

- **Confusion Matrix:**
  
  \begin{bmatrix}
  3478 & 270 \\
  1059 & 708 \\
  \end{bmatrix}
  

- **Classification Report:**

          precision    recall  f1-score   support

       0       0.77      0.93      0.84      3748
       1       0.72      0.40      0.52      1767

       accuracy                    0.76      5515



The model was evaluated on a dataset with 22,058 observations, and the performance metrics provide insights into the model's predictive capabilities.

## Limitations

The model has several limitations, including:

- **Assumption of Linearity:** Logistic Regression assumes a linear relationship between the log-odds of the dependent variable and the independent variables.

- **Sensitivity to Outliers:** Logistic Regression can be sensitive to outliers in the data.

- **Multicollinearity:** High correlations among features may impact the stability of coefficient estimates.

- **Assumption of Independence:** Logistic Regression assumes that observations are independent, and violations of this assumption may affect the validity of results.

## Trade-offs

The trade-offs of using Logistic Regression for binary classification include:

- **Interpretability:** Logistic Regression provides interpretable coefficients, allowing understanding of the impact of each feature on the log-odds of the outcome.

- **Assumption Strictness:** Logistic Regression has assumptions that need to be satisfied for valid results, and careful consideration of these assumptions is necessary.

- **Simplicity:** Logistic Regression is a relatively simple model, which may not capture complex non-linear relationships.

- **Threshold Selection:** The choice of the classification threshold may impact the balance between precision and recall.

Consideration of these trade-offs is crucial when deciding to use Logistic Regression for binary classification with the specified features for a particular analysis.

## Model Description K-Nearest Neighbors (KNN)

**Input:** The model uses the K-Nearest Neighbors (KNN) algorithm with varying numbers of neighbors.

**Output:** The model predicts binary outcomes and provides evaluation metrics such as RMSE (Root Mean Squared Error) and Misclassification Rate. The best-performing configuration is highlighted.

**Model Architecture:** KNN algorithm with different numbers of neighbors.

## Performance

The model's performance is assessed for different numbers of neighbors:

- **Number of Neighbors: 2**
  - RMSE: 0.5108
  - Misclassification Rate: 0.2609

- **Number of Neighbors: 3**
  - RMSE: 0.5119
  - Misclassification Rate: 0.2620

- **Number of Neighbors: 4**
  - RMSE: 0.5019
  - Misclassification Rate: 0.2519

- **Number of Neighbors: 5**
  - RMSE: 0.4973
  - Misclassification Rate: 0.2473

- **Number of Neighbors: 6**
  - RMSE: 0.4877
  - Misclassification Rate: 0.2379

- **Number of Neighbors: 7**
  - RMSE: 0.4909
  - Misclassification Rate: 0.2410

- **Number of Neighbors: 8**
  - RMSE: 0.4861
  - Misclassification Rate: 0.2363

- **Number of Neighbors: 9**
  - RMSE: 0.4896
  - Misclassification Rate: 0.2397

- **Number of Neighbors: 10**
  - RMSE: 0.4851
  - Misclassification Rate: 0.2354 (Best Performing)

- **Number of Neighbors: 11**
  - RMSE: 0.4868
  - Misclassification Rate: 0.2370

**Best Configuration:**
- Best Number of Neighbors: 10
- Best RMSE: 0.4851
- Best Misclassification Rate: 0.2354

### Model Evaluation Metrics:

- **Confusion Matrix:**
  
  \begin{bmatrix}
  3368 & 380 \\
  918 & 849 \\
  \end{bmatrix}
  

- **Classification Report:**

          precision    recall  f1-score   support

       0       0.79      0.90      0.84      3748
       1       0.69      0.48      0.57      1767

       accuracy                    0.76      5515

The model was evaluated on a dataset with 5,515 observations, and the performance metrics provide insights into the model's predictive capabilities.

## Limitations

The KNN algorithm has some limitations, including:

- **Sensitivity to Scale:** The algorithm is sensitive to the scale of features, requiring normalization.

- **Computational Intensity:** Prediction time may be high for large datasets.

- **Choice of K:** The performance may vary with the choice of the number of neighbors (K).

## Trade-offs

The trade-offs of using KNN include:

- **Simplicity:** KNN is a simple and intuitive algorithm.

- **Lack of Model Interpretability:** KNN lacks clear interpretability compared to some other models.

- **Dependency on Training Data:** Performance is heavily influenced by the characteristics of the training data.

- **Impact of Outliers:** Outliers can significantly affect predictions.

Consideration of these trade-offs is crucial when deciding to use KNN for a particular analysis.

## Model Description Random Forest 

**Input:** The model uses the Random Forest algorithm for binary classification.

**Output:** The model predicts binary outcomes and provides evaluation metrics such as a confusion matrix, classification report, RMSE (Root Mean Squared Error), and Misclassification Rate.

**Model Architecture:** Random Forest algorithm.

## Performance

### Model Evaluation Metrics:

- **Confusion Matrix:**
  
  \begin{bmatrix}
  3274 & 474 \\
  834 & 933 \\
  \end{bmatrix}
  

- **Classification Report:**

          precision    recall  f1-score   support

       0       0.80      0.87      0.83      3748
       1       0.66      0.53      0.59      1767

       accuracy                    0.76      5515
    

- **RMSE (Root Mean Squared Error):** 0.4870024136093038
- **Misclassification Rate:** 0.23717135086128738
- **Accuracy:** 0.7628
- **Precision:** 0.6631
- **Recall:** 0.5280

The model was evaluated on a dataset with 5,515 observations, and the performance metrics provide insights into the model's predictive capabilities.

## Limitations

The Random Forest algorithm has some limitations, including:

- **Lack of Interpretability:** Random Forest models are less interpretable compared to simpler models.

- **Computational Intensity:** Training and prediction time may be higher compared to simpler models.

- **Overfitting:** Random Forests can overfit noisy data.

## Trade-offs

The trade-offs of using Random Forest include:

- **High Predictive Power:** Random Forests often have high predictive accuracy.

- **Ensemble of Trees:** By combining multiple decision trees, Random Forests reduce the risk of overfitting present in individual trees.

- **Feature Importance:** Random Forests provide insights into feature importance.

- **Model Complexity:** Random Forests can be complex, and understanding the impact of individual features may be challenging.

Consideration of these trade-offs is crucial when deciding to use Random Forest for a particular analysis.

## Random Forest Classifier for Binary Classification with Hyperparameter Tuning

### Model Description

**Input:** The model utilizes a Random Forest classifier for binary classification.

**Output:** The model predicts binary outcomes and has undergone hyperparameter tuning using GridSearchCV.

**Model Hyperparameters:**
```python
{
    'classifier__max_depth': 10,
    'classifier__max_features': 'auto',
    'classifier__min_samples_leaf': 2,
    'classifier__min_samples_split': 2,
    'classifier__n_estimators': 150
}

```

#### max_depth (Maximum Depth of Trees):

- **Value:** 10
- **Explanation:** Controls the maximum depth of decision trees. Higher values allow capturing more complex patterns but increase the risk of overfitting.

#### max_features (Maximum Number of Features for Split):

- **Value:** 'auto'
- **Explanation:** Determines the maximum number of features considered for splitting a node. 'auto' considers all features, a common choice.

#### min_samples_leaf (Minimum Samples in Leaf Node):

- **Value:** 2
- **Explanation:** Sets the minimum samples required in a leaf node. Lower values increase sensitivity to noise and capture more fine-grained patterns, controlling overfitting.

#### min_samples_split (Minimum Samples for Node Split):

- **Value:** 2
- **Explanation:** Sets the minimum samples required to split an internal node. Similar to `min_samples_leaf`, it controls tree size and prevents overfitting.

#### n_estimators (Number of Trees in the Forest):

- **Value:** 150
- **Explanation:** Determines the number of decision trees in the Random Forest. More trees generally improve performance but increase computational cost.

#### Performance

The model's performance metrics, such as accuracy, precision, recall, and confusion matrix, were evaluated using a test dataset.

- **Confusion Matrix:**
  
  \begin{bmatrix}
  3329 & 419 \\
  834 & 933 \\
  \end{bmatrix} 
  

#### Limitations

- **Computational Intensity:** Random Forest models with a large number of trees can be computationally intensive.

#### Trade-offs

- **Accuracy vs. Computational Cost:** The selected hyperparameters balance accuracy with computational cost.


**Best Hyperparameters:**

The best hyperparameters obtained from Bayesian optimization for the Random Forest classifier are represented as an OrderedDict:

```python
\[
\text{OrderedDict}\left(\left[
    ('classifier__max_depth', 12),
    ('classifier__max_features', 0.1),
    ('classifier__min_samples_leaf', 1),
    ('classifier__min_samples_split', 6),
    ('classifier__n_estimators', 150)
\right]\right)
\]
```
Let's break down the meaning of each hyperparameter:

1. **`max_depth` (Maximum Depth of Trees):**
   - **Value:** 12
   - **Explanation:** Controls the maximum depth of decision trees. Higher values allow capturing more complex patterns but increase the risk of overfitting.

2. **`max_features` (Maximum Fraction of Features for Split):**
   - **Value:** 0.1
   - **Explanation:** Represents the fraction of features to consider for splitting a node. A value of 0.1 means 10% of features will be considered for each split.

3. **`min_samples_leaf` (Minimum Samples in Leaf Node):**
   - **Value:** 1
   - **Explanation:** Sets the minimum number of samples required in a leaf node. Lower values increase sensitivity to noise and capture more fine-grained patterns, controlling overfitting.

4. **`min_samples_split` (Minimum Samples for Node Split):**
   - **Value:** 6
   - **Explanation:** Sets the minimum number of samples required to split an internal node. Similar to `min_samples_leaf`, it controls tree size and prevents overfitting.

5. **`n_estimators` (Number of Trees in the Forest):**
   - **Value:** 150
   - **Explanation:** Determines the number of decision trees in the Random Forest. More trees generally improve performance but increase computational cost.
   
#### Performance

The model's performance metrics, such as accuracy, precision, recall, and confusion matrix, were evaluated using a test dataset.

- **Confusion Matrix:**
  
  \begin{bmatrix}
  3339 & 409 \\
  847 & 920 \\
  \end{bmatrix} 
 

These hyperparameters were selected based on the optimization process to maximize the performance of the Random Forest classifier.


