🩺 Diabetes Prediction Using Machine Learning

This project focuses on building and evaluating machine learning models to predict the likelihood of diabetes in individuals using medical diagnostic data. The goal is to support early detection by leveraging predictive analytics.

📘 Overview

Diabetes is a growing global health concern, and timely diagnosis plays a crucial role in preventing complications. Using a dataset that includes health metrics such as glucose levels, BMI, insulin, and age, we explore multiple machine learning approaches to predict whether a patient is likely to be diabetic.

📁 Project Files
**`Task1_Diabetes_prediction.ipynb`** – The main Jupyter notebook containing:

  * Data preprocessing
  * Model building
  * Performance evaluation

📊 Dataset Description

We use the **Pima Indians Diabetes Dataset**, which contains the following features:

| Feature                  | Description                                    |
| ------------------------ | ---------------------------------------------- |
| Pregnancies              | Number of pregnancies                          |
| Glucose                  | Plasma glucose concentration                   |
| BloodPressure            | Diastolic blood pressure (mm Hg)               |
| SkinThickness            | Triceps skin fold thickness (mm)               |
| Insulin                  | 2-Hour serum insulin (mu U/ml)                 |
| BMI                      | Body mass index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction | Diabetes pedigree function                     |
| Age                      | Age in years                                   |
| Outcome                  | Class label (1: diabetic, 0: non-diabetic)     |

---

## 🔧 Workflow Summary

### 1. **Data Preprocessing**

* Handled missing or zero values
* Normalized feature values using `MinMaxScaler`
* Split data into training and testing sets

### 2. **Model Development**

* Evaluated multiple algorithms:

  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Classifier
  * K-Nearest Neighbors (KNN)
  * Support Vector Machine (SVM)

### 3. **Model Evaluation**

* Metrics used:

  * Accuracy Score
  * Confusion Matrix
  * ROC-AUC Curve
  * Classification Report

Visualizations were included to compare model performance and interpret results.

🚀 How to Run the Project

1. Download or clone this repository.
2. Make sure the following Python libraries are installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. Open the notebook in Jupyter or Google Colab.
4. Run all cells to reproduce the results.

---

## 📌 Results

Each model was evaluated on standard performance metrics. The best-performing model(s) were highlighted based on their ability to generalize well to unseen data. Please refer to the final section of the notebook for a detailed comparison.
