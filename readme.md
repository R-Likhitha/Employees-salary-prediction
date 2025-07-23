# 💼 Employee Salary Prediction using Machine Learning

This project is a complete machine learning pipeline and interactive web app that predicts whether an employee’s salary is **greater than or less than $50K** based on key attributes such as **age, workclass, occupation, capital gain, hours per week**, and **native country**.

Built using `scikit-learn`, `pandas`, and `Streamlit`, this application demonstrates the power of data preprocessing, model optimization, and real-time prediction in a clean user interface.

---

## 📊 Problem Statement

Predict whether an individual earns **>50K or <=50K ** . This is a **binary classification** problem solved using a **Random Forest** model trained on a subset of relevant features.

---

## 🎯 Features Used

| Feature         | Description                                      |
|----------------|--------------------------------------------------|
| `age`           | Age of the individual                            |
| `occupation`    | Type of occupation                               |
| `workclass`     | Type of employment (e.g., private, gov, etc.)    |
| `capital-gain`  | Capital gains earned                             |
| `hours-per-week`| Number of working hours per week                 |
| `native-country`| Country of residence                             |

These features were selected for their predictive power and relevance to salary classification.

---

## 🧠 Technologies & Concepts

### ✅ Machine Learning
- **Random Forest Classifier**: Used for classification due to its high accuracy and robustness.
- **GridSearchCV**: For optimizing model hyperparameters.
- **Pipeline**: Combines preprocessing and modeling for cleaner code and easier deployment.

### ✅ Data Preprocessing
- **ColumnTransformer**:
  - `StandardScaler` for numeric data
  - `OneHotEncoder` for categorical variables
- Missing values handled and cleaned
- Proper label encoding for training

### ✅ Deployment
- **Streamlit** used to build an intuitive web UI
- Model saved with `joblib` (`salary_model.pkl`)
- Features order saved as `feature_columns.pkl`

---

