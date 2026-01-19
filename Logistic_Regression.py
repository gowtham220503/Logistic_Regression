
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

df = pd.read_csv("diabetes.csv")

#Data Exploration
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nHead:\n", df.head())
print("\nInfo:\n")
print(df.info())

df.describe()

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.title("Outcome Distribution (0 = Non-diabetic, 1 = Diabetic)")
plt.show()

numeric_cols = df.columns.drop("Outcome")
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="Outcome", y=col, data=df)
    plt.title(f"{col} by Outcome")
    plt.xlabel("Outcome (0 = Non-diabetic, 1 = Diabetic)")
    plt.ylabel(col)
    plt.show()

corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()

# Higher Glucose, BMI, DiabetesPedigreeFunction, and Age values appear more common among diabetic patients (Outcome = 1) than non-diabetics. Pregnancies also tend to be higher in the diabetic group, reflecting age and parity effects. Some physiological variables like Glucose, BloodPressure, SkinThickness, Insulin, and BMI contain zeros, suggesting missing or unrecorded measurements that should be treated carefully before modeling. Relationships between predictors themselves are moderate at best, so multicollinearity is limited, and each feature can add incremental information for diabetes prediction

#Data Preprocessing
df_impute = df.copy()
missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in missing:
    median_value = df_clean[col].median()
    df_clean[col].fillna(median_value, inplace=True)
    print(f"{col}: imputed with median = {median_value}")

df_final = df_imputed_by_class.copy()
# Inspect dtypes and unique counts
print(df_final.dtypes)
print("\nUnique values per column:")
for col in df_final.columns:
    print(col, "->", df_final[col].nunique())

X = df_clean.drop("Outcome", axis=1)
y = df_clean["Outcome"]
y = y.astype(int)

print(y.value_counts())
print(y.dtypes)

#create a categorical age band
age_bins = [20, 30, 40, 50, 60, 80]
age_labels = ["20-29", "30-39", "40-49", "50-59", "60+"]
X["AgeGroup"] = pd.cut(X["Age"], bins=age_bins, labels=age_labels, right=False)

# One-hot encode AgeGroup
X = pd.get_dummies(X, columns=["AgeGroup"], drop_first=True)
print(X.head())

#Model Building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, solver="lbfgs")
log_reg.fit(X_train, y_train)

from sklearn.preprocessing import StandardScaler
numeric_features = X.columns  
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
])
pipeline.fit(X_train, y_train)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Accuracy     : {acc:.3f}")
print(f"Precision    : {prec:.3f}")
print(f"Recall       : {rec:.3f}")
print(f"F1-score     : {f1:.3f}")
print(f"ROC-AUC score: {roc_auc:.3f}")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

#Visualize ROC curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#Interpretation
clf = pipeline.named_steps["clf"]

feature_names = X.columns
coefs = clf.coef_[0]

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefs
}).sort_values("Coefficient", ascending=False)

print(coef_df)

# In this logistic regression model, features contribute differently to predicting diabetes (survival vs. non‑survival analogy). Glucose typically emerges as the most important predictor, with higher values strongly increasing the probability of a positive Outcome. BMI and Age also show substantial positive effects, indicating that obesity and older age are associated with higher diabetes risk. Pregnancies often has a moderate positive coefficient, reflecting slightly higher risk with more pregnancies, consistent with gestational diabetes links. DiabetesPedigreeFunction captures genetic predisposition; higher values increase the odds of diabetes, highlighting the role of family history. BloodPressure, SkinThickness and Insulin can be informative but sometimes have smaller or unstable coefficients due to missingness and collinearity, so their individual impact may appear limited in the model. Overall, glucose, BMI, age and hereditary factors dominate risk prediction, while other clinical measures provide secondary refinements to the probability estimates.

