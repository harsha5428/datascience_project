# README - Breast Cancer Detection Project

## Project Overview
This project focuses on the classification of breast cancer using two datasets:
1. **Breast Cancer Wisconsin (Diagnostic) Dataset**
2. **Mammographic Mass Dataset**
The goal is to predict whether a tumor is benign or malignant by applying machine learning techniques and comparing models trained on various dataset variations.

## Datasets
- **[Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))** :
Contains 30 numerical features computed from digitized images of breast masses.
- **[Mammographic Mass Dataset](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)** :
Contains 6 features describing the characteristics of detected masses.


## Objective

- Develop machine learning models to classify tumors.
- Apply preprocessing and exploratory data analysis (EDA) to both datasets.
- Train and evaluate four machine learning models:
  - **Random Forest (RF)**
  - **Support Vector Machine (SVM)**
  - **Gradient Boosting (GB)**
  - **Logistic Regression (LR)**
- Perform hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
- Address class imbalance using SMOTE and RFE.
- Use the best-performing models in a voting classifier.

## Workflow

### Wisconsin Dataset

1. **Models on Imbalanced Data (Baseline Model)**:
   - Trained models on imbalanced data without addressing class imbalance.
2. **Hyperparameter Tuning (Imbalanced Data)**:
   - Hyperparameter tuning techniques were applied to optimize model performance.
3. **RFE with Imbalanced Data**:
   - Recursive Feature Elimination (RFE) was applied to select the top 10 most important features.
4. **RFE + Hyperparameter Tuning (Imbalanced Data)**:
   - Hyperparameter tuning techniques were applied to the RFE-transformed dataset to further enhance model accuracy.
5. **SMOTE with Models (Baseline)**:
   - Applied Hyperparameter tuning SMOTE to balance the dataset.
6. **SMOTE + Hyperparameter Tuning (Random Forest)**:
   - Hyperparameter tuning was used on the SMOTE-balanced data.
7. **SMOTE + RFE (Random Forest)**:
   - Applied RFE after balancing data with SMOTE.
8. **SMOTE + RFE + Hyperparameter Tuning**:
   - Combined RFE with SMOTE and applied hyperparameter tuning for optimal performance.

### Mammographic Mass Dataset

1. **Models on Imbalanced Data (Baseline Model)**:
   - Trained models on imbalanced data without addressing class imbalance.
2. **Hyperparameter Tuning (Imbalanced Data)**:
   - Hyperparameter tuning techniques were applied to fine-tune model performance. 
3. **SMOTE with Models (Baseline)**:
   - Applied SMOTE to balance the dataset.
4. **SMOTE + Hyperparameter Tuning**:
   - Hyperparameter tuning was used to optimize models trained on SMOTE-balanced data.



### Evaluation Metrics

- **Classification Report**: Provides precision, recall, F1-score, and accuracy.
- **Confusion Matrix**: Visualizes true positives, true negatives, false positives, and false negatives.
- **Learning Curve**: Demonstrates the model's performance across different training sizes.
- **ROC Curve**: Evaluates the trade-off between sensitivity and specificity.

## Model Selection and Voting Classifier

- The best-performing models from each variation were selected.
- A voting classifier was built using these models to improve overall performance and robustness.

### Best Models for Voting Classifier

- **Random Forest (Wisconsin)**:

  - **Model**: Random Forest with RFE and SMOTE (After Hyperparameter Tuning)
  - **Accuracy**: 96%
  - **AUC**: 1.00
  - **Confusion Matrix**: 2 false positives, 2 false negatives
  - **Summary**: SMOTE balanced the dataset, and RFE ensured the most important features were selected, enhancing both performance and interpretability.

- **Support Vector Machine (Wisconsin)**:

  - **Model**: SVM with RFE on SMOTE Balanced Data(After Hyperparameter Tuning)
  - **Accuracy**: 95%
  - **AUC**: 1.00
  - **Confusion Matrix**: 3 false positives, 3 false negatives
  - **Summary**: Strong classification with high precision (0.96 for benign, 0.93 for malignant) and excellent recall.

- **Gradient Boosting (Wisconsin)**:

  - **Model**: Gradient Boosting with RFE and SMOTE(After Hyperparameter Tuning)
  - **Accuracy**: 97%
  - **AUC**: 1.00
  - **Confusion Matrix**: 1 false negative, 2 false positives
  - **Summary**: Near-perfect classification with balanced precision, recall, and F1-scores.

- **Logistic Regression (Wisconsin)**:

  - **Model**: Logistic Regression with RFE and SMOTE(After Hyperparameter Tuning)
  - **Accuracy**: 97%
  - **AUC**: 1.00
  - **Confusion Matrix**: 1 false positive, 2 false negatives
  - **Summary**: Excellent precision and recall, with minimal misclassification.

- **Random Forest (Mammographic Mass)**:

  - **Model**: RF on SMOTE Balanced Data(After Hyperparameter Tuning)
  - **Accuracy**: 83%
  - **AUC**: 0.88
  - **Confusion Matrix:** 21 false positive,11 false negatives
  - **Summary**: Balanced performance with strong classification capabilities.

- **SVM (Mammographic Mass)**:

  - **Model**: SVM on SMOTE Balanced Data(After Hyperparameter Tuning)
  - **Accuracy**: 80%
  - **AUC**: 0.88
  - **Confusion Matrix:** 28 false positive, 10 false negatives
  - **Summary**: Strong recall for malignant cases, reliable for identifying malignancies.

- **Gradient Boosting (Mammographic Mass)**:

  - **Model**: GB on Imbalanced Data(After Hyperparameter Tuning)
  - **Accuracy**: 83%
  - **AUC**: 0.89
  - **Confusion Matrix:** 21 false positive, 11 false negatives
  - **Summary**: Superior accuracy in imbalanced conditions, strong classification performance.

- **Logistic Regression (Mammographic Mass)**:

  - **Model**: LR on Imbalanced Data(After Hyperparameter Tuning)
  - **Accuracy**: 83%
  - **AUC**: 0.90
  - **Confusion Matrix:** 17 false positive, 11 false negatives
  - **Summary**: Effective and reliable model for the mammographic mass dataset.
### Voting Classifier
- ****Voting Classifier (Wisconsin)**
- **Model**: Voting Classifier
  - **Accuracy**: 97%
  - **AUC**: 1
  - **Confusion Matrix:** 1 false positive, 1 false negatives
    
- **Voting Classifier (Mammographic Mass)**:
  - **Model**: Voting Classifier
  - **Accuracy**: 84%
  - **AUC**: 0.89
  - **Confusion Matrix:** 20 false positive, 11 false negatives

## Requirements

- Python 3.3
- **Core Libraries**: numpy, pandas, random
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Model Evaluation**: roc_auc_score, classification_report
- **Pipeline/Feature Selection**: Pipeline, RFE
  
## How to Run
1. Clone the repository.
2. Install required dependencies.
3. Run the Jupyter notebook or Python script.

## Acknowledgments
- Datasets sourced from UCI Machine Learning Repository.
