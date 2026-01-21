import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------------
# LOAD DATASET
# ---------------------------------
df = pd.read_csv("data/employee_attrition.csv")

# Encode target variable
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Split features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ---------------------------------
# FEATURE TYPES
# ---------------------------------
categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

# ---------------------------------
# PREPROCESSING
# ---------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ---------------------------------
# MODEL
# ---------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ---------------------------------
# PIPELINE
# ---------------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ---------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------
# TRAIN MODEL
# ---------------------------------
pipeline.fit(X_train, y_train)

# ---------------------------------
# PREDICTIONS
# ---------------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ---------------------------------
# EVALUATION METRICS
# ---------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=["No Attrition", "Attrition"]
)

# ---------------------------------
# PRINT RESULTS
# ---------------------------------
print("\n================ MODEL EVALUATION ================\n")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-score      : {f1:.4f}")
print(f"ROC-AUC Score : {roc_auc:.4f}\n")

print("Classification Report:\n")
print(report)

print("Confusion Matrix:\n")
print(cm)

# ---------------------------------
# SAVE MODEL & METRICS
# ---------------------------------
joblib.dump(pipeline, "model/attrition_model.pkl")

with open("model/evaluation_metrics.txt", "w") as f:
    f.write("MODEL EVALUATION METRICS\n\n")
    f.write(f"Accuracy      : {accuracy:.4f}\n")
    f.write(f"Precision     : {precision:.4f}\n")
    f.write(f"Recall        : {recall:.4f}\n")
    f.write(f"F1-score      : {f1:.4f}\n")
    f.write(f"ROC-AUC Score : {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nModel trained and saved successfully.")
