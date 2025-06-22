import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("cardio_train.csv", sep=';')
# Convert age to years
df["age_years"] = (df["age"] / 365).astype(int)

# Compute BMI = weight (kg) / height (m)^2
df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)

# Bin age into 4 categories
df["age_group"] = pd.cut(
    df["age_years"],
    bins=[0, 39, 49, 59, 100],
    labels=["<40", "40-49", "50-59", "60+"]
)

# Drop original columns
df = df.drop(columns=["id", "age", "age_years", "height", "weight"])

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=["age_group", "cholesterol", "gluc"], drop_first=True)

# Split features and target
X = df_encoded.drop("cardio", axis=1)
y = df_encoded["cardio"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pipeline: scale + logistic regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save model
joblib.dump(model, "logreg_model.joblib")

