# Supervised-learning

## Assignment Overview

This assignment involved applying various machine learning models to two main tasks:
1.  **Classification:** Predicting dog breeds from images.
2.  **Regression:** Predicting bone age from images and associated data.

For each task, multiple models were implemented, tuned, and evaluated. The process involved data preprocessing, hyperparameter optimization using techniques like GridSearchCV and RandomizedSearchCV, k-fold cross-validation, and analysis of performance metrics.

## Datasets Used

1.  **Classification (Dog Breeds):**
    *   Source: [Kaggle Dog Breed Image Dataset](https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset)
    *   Details: 10 folders, each named by dog breed, containing a total of 967 images.
2.  **Regression (Bone Age):**
    *   Source: [Kaggle RSNA Bone Age Dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age/data?select=boneage-training-dataset)
    *   Details: 629 images initially, with a CSV file containing `id`, `boneage`, and `male` (gender) columns. After filtering for valid images, 558 samples were used for most regression models. For Polynomial Regression, a smaller subset (50 images) was used due to memory constraints.

## General Methodology

*   **Preprocessing:**
    *   Image resizing to consistent dimensions (e.g., 64x64).
    *   Image flattening into 1D arrays.
    *   Label Encoding for categorical target variables (dog breeds).
    *   Feature Scaling (e.g., `StandardScaler`).
    *   Handling missing/invalid data (e.g., filtering image IDs).
*   **Model Training & Evaluation:**
    *   Splitting data into training and testing sets.
    *   K-Fold Cross-Validation (e.g., `KFold`, `StratifiedKFold`) for robust evaluation.
    *   Hyperparameter Tuning: `GridSearchCV`, `RandomizedSearchCV`.
    *   Handling Class Imbalance (for classification): SMOTE, `class_weight='balanced'`.
*   **Error Handling & Debugging:**
    *   Addressing issues like mismatched data shapes, numerical conversion errors, long run times, and model convergence problems.
    *   Adjusting parameters, reducing dataset/parameter grid size for debugging.
    *   Using `zero_division=0` in metric calculations.
    *   Implementing early stopping for Decision Trees.

---

## I. Classification Results (Dog Breeds)

The goal was to classify images into 10 different dog breeds.

### 1. Logistic Regression

*   **Challenges & Fixes:**
    *   Filenames (breed names) needed mapping to integers.
    *   Images of different shapes: Resized and flattened.
    *   Smallest class size issues with `n_splits`: Adjusted `n_splits`, used SMOTE for oversampling, `class_weight='balanced'`, and `zero_division=0`.
*   **Hyperparameters (Best):** `{'C': 100}` (with `max_iter=1000`, `class_weight='balanced'`)
*   **Cross-Validation:** `StratifiedKFold(n_splits=5)`
*   **Performance Metrics:**
    *   Precision: 0.990
    *   Recall: 0.991
    *   F1 Score: 0.990
*   **Confusion Matrix:** Indicated very high accuracy with few misclassifications.

### 2. Support Vector Machine (SVM)

*   **Hyperparameters (Best):** `{'C': 10, 'kernel': 'rbf', 'gamma': 'scale'}`
    *   `C=10`: Moderate regularization.
    *   `kernel='rbf'`: Suitable for non-linear data.
    *   `gamma='scale'`: Adjusts influence based on input feature variance.
*   **Cross-Validation:** `KFold(n_splits=5, shuffle=True)`
*   **Performance Metrics:**
    *   Precision: 0.991
    *   Recall: 0.990
    *   F1 Score: 0.990
*   **Confusion Matrix:** Showed very few misclassifications, indicating accurate class predictions.

### 3. Decision Tree

*   **Challenges & Fixes:**
    *   Initial long run times: Reduced hyperparameter search space for debugging.
    *   Optimization: Used `RandomizedSearchCV` instead of `GridSearchCV`, `n_jobs=-1` for parallel processing, and implemented early stopping.
*   **Hyperparameters (Best):** `{'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 5}`
*   **Performance Metrics (After Optimization):**
    *   Precision: 0.959
    *   Recall: 0.928
    *   F1 Score: 0.936 (Balanced F1)
*   **Confusion Matrix:** Performed well for most classes, with some misclassifications noted for classes 0 and 2.

### 4. Multi-layer Perceptron (MLP)

*   **Challenges & Fixes:**
    *   Initial script stuck during `GridSearchCV`: Reduced parameter grid and added `verbose=2` for progress monitoring.
*   **Hyperparameters (Best - Expanded Grid):** `{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50), 'learning_rate': 'constant'}`
    *   `activation='relu'`: ReLU for hidden layers.
    *   `alpha=0.0001`: L2 penalty for regularization.
    *   `hidden_layer_sizes=(50, 50)`: Two hidden layers, each with 50 neurons.
*   **Performance Metrics:**
    *   Precision: 0.9908
    *   Recall: 0.9897
    *   F1 Score: 0.9899
*   **Confusion Matrix:** High accuracy, e.g., Class 2 had 25 correct, 1 misclassification (as Class 4).

### 5. Random Forest Classifier

*   **Challenges & Fixes:**
    *   Long run time with extensive parameter grid: Reduced parameter grid.
*   **Hyperparameters (Best):** `{'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}`
    *   `max_depth=None`: Nodes expanded until pure or min samples reached.
    *   `min_samples_leaf=2`: Prevents overfitting.
    *   `min_samples_split=2`: Min samples to split a node.
    *   `n_estimators=200`: 200 trees in the forest.
*   **Performance Metrics:**
    *   Precision: 0.9903
    *   Recall: 0.9897
    *   F1 Score: 0.9897
*   **Confusion Matrix:** Most classes predicted correctly with few misclassifications (e.g., Class 2: 25 correct, 1 misclassified as Class 4).

---

## II. Regression Results (Bone Age Prediction)

The goal was to predict the numerical bone age.

### 1. Linear Regression

*   **Challenges & Fixes:**
    *   Initial `ValueError` due to empty training set from image ID mismatch: Handled ID range and ensured paths exist.
*   **Hyperparameters (Best):** `{'copy_X': True, 'fit_intercept': True}`
*   **Mean Squared Error (MSE):** 3874.23
*   **Scatter Plot (True vs. Predicted):** Showed a positive correlation but with significant variability in prediction accuracy.
*   **Residual Plot:** Indicated the model was unbiased with no clear pattern or trend in residuals, which is generally good. However, significant prediction errors were present for some cases.

### 2. Polynomial Regression

*   **Challenges & Fixes:**
    *   Excessive memory usage: Reduced number of images to 50.
    *   Complexity: Set a fixed `degree=2` and `cv=3`.
*   **Hyperparameters (Best):** `{'regression__fit_intercept': True}` (within a pipeline)
*   **Mean Squared Error (MSE):** 3185.64
*   **Scatter Plot (True vs. Predicted):** Generally performed reasonably well, with predictions often close to true values. Outliers and scatter indicated potential for significant errors. Small sample size limited generalization.
*   **Residual Plot:** Model appeared unbiased, but with significant prediction errors for some cases. The pattern of residuals was not clear due to the small sample.

### 3. Support Vector Regressor (SVR)

*   **Hyperparameters (Best):** `{'svr__C': 10, 'svr__epsilon': 0.3, 'svr__kernel': 'rbf'}` (within a pipeline)
*   **Mean Squared Error (MSE):** 2032.83
*   **Scatter Plot (True vs. Predicted):** Positive correlation with true ages but exhibited considerable variability. Accuracy seemed to decrease for higher bone ages.
*   **Residual Plot:** Points scattered around the zero line, indicating no systematic over or under-prediction. No clear trend or curve, and the spread of residuals seemed fairly consistent.

### 4. Multi-layer Perceptron (MLP) Regressor

*   **Challenges & Fixes:**
    *   Initial "Maximum iterations reached and optimization hasn't converged" error: Simplified parameter grid and increased verbosity.
*   **Hyperparameters (Best):** `{'mlp__activation': 'relu', 'mlp__alpha': 0.001, 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'constant'}` (within a pipeline)
*   **Mean Squared Error (MSE):** 6558.30
*   **Scatter Plot (True vs. Predicted):** Many points clustered around the perfect prediction line. The spread seemed to increase for higher bone ages, suggesting less accuracy for older individuals.
*   **Residual Plot:** Showed a noticeable pattern (heteroscedasticity) – the spread of residuals decreased as predicted bone age increased. Indicated underestimation for younger subjects (0-50 months) and mild overestimation for older subjects (100+ months).

### 5. Random Forest Regressor

*   **Noted as achieving the lowest Mean Squared Error among the regression models.**
*   **Hyperparameters (Best):** `{'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 149}`
*   **Mean Squared Error (MSE):** 1779.20
*   **Scatter Plot (True vs. Predicted):** Spread seemed relatively consistent across different ages. No obvious bias towards over or under-prediction for specific age ranges.
*   **Residual Plot:** No clear trend or curve as predicted bone age increased. The spread of residuals seemed fairly constant.

---

## Reflections & Learnings


*   **Most Difficult Part:**
    *   Tuning hyperparameters for each estimator, determining appropriate parameters and values.
    *   Managing long run times and adjusting iteration counts/parameters accordingly.
*   **Most Rewarding Part:**
    *   Practicing with 5 different models for both classification and numerical data tasks.
*   **Key Learnings:**
    *   Strategies for dealing with excessive memory usage and long run times.
    *   Adjusting parameters in grid search effectively.
    *   Utilizing preprocessed data and label arrays to avoid redundant processing when testing different models.


## Resources Used

*   **Scikit-learn Documentation:**
    *   KFold: [sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
    *   Train-Test Split: [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    *   GridSearchCV: [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
*   **Datasets:** (Links provided in the "Datasets Used" section above)
