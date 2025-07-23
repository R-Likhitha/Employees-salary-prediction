# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

print("📥 Loading data...")
df = pd.read_csv("adult 3.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '-')

# Select required columns + target
required_cols = ['age', 'occupation', 'capital-gain', 'hours-per-week', 'native-country', 'workclass', 'income']
df = df[required_cols]
df = df.replace('?', np.nan).dropna()

print("✅ Cleaned data with selected features.")

# Define X and y
X = df.drop("income", axis=1)
y = df["income"]

# Identify types
numeric_features = ['age', 'capital-gain', 'hours-per-week']
categorical_features = ['occupation', 'native-country', 'workclass']

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# GridSearchCV
param_grid = {
    "classifier__n_estimators": [100],
    "classifier__max_depth": [None, 10]
}

print("⚙️ Training model...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

# Save model and feature columns
joblib.dump(grid_search.best_estimator_, "salary_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("✅ Model training complete and saved!")
print("🎯 Training Accuracy:", accuracy_score(y, grid_search.predict(X)))
