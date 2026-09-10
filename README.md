# Logistic Regression – Diabetes Prediction

## Project Overview

This project uses **Logistic Regression** to predict whether a person is likely to have diabetes based on medical information.

The model uses patient-related features such as glucose level, blood pressure, BMI, age, insulin, and other health measurements to predict the **Outcome** as:

* `0` – No Diabetes
* `1` – Diabetes

## Objective

The main objective of this project is to build a simple machine learning classification model that can predict diabetes outcomes from medical data.

## Dataset

The dataset contains **768 records** and **9 columns**.

### Features

| Feature                  | Description                |
| ------------------------ | -------------------------- |
| Pregnancies              | Number of pregnancies      |
| Glucose                  | Glucose level              |
| BloodPressure            | Blood pressure             |
| SkinThickness            | Skin thickness             |
| Insulin                  | Insulin level              |
| BMI                      | Body Mass Index            |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age                      | Age of the person          |
| Outcome                  | Target variable            |

## Project Steps

1. Import required Python libraries.
2. Load the diabetes dataset.
3. Check dataset information.
4. Check for missing values.
5. Perform basic statistical analysis.
6. Separate input features `X` and target `y`.
7. Split the data into training and testing sets.
8. Train a Logistic Regression model.
9. Make predictions on test data.
10. Evaluate the model using a confusion matrix, accuracy, and classification report.

## Machine Learning Model

### Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for **classification problems**.

In this project:

* Input: Patient medical information
* Output: Diabetes prediction (`0` or `1`)
* Test size: 20%
* Random state: 32

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

## Model Evaluation

The Logistic Regression model achieved:

**Accuracy: 78.57%**

### Confusion Matrix

```text
[[85 14]
 [19 36]]
```

### Classification Report

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| 0     |      0.82 |   0.86 |     0.84 |
| 1     |      0.72 |   0.65 |     0.69 |

## Additional Experiment

The notebook also contains an experiment using **Linear Regression** on the same dataset.

The Linear Regression model was evaluated using:

* Mean Squared Error (MSE): `0.1422`
* Mean Absolute Error (MAE): `0.3178`
* Root Mean Squared Error (RMSE): `0.3771`
* R² Score: `0.3807`

The Linear Regression predictions were then passed through a **sigmoid function** and converted into binary classes using a calculated threshold.

The manually generated classification results achieved approximately **77% accuracy**.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook




