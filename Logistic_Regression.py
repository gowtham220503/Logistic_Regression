import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

st.title("Logistic Regression Streamlit App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # ------------------ Boxplot ------------------
    st.subheader("Boxplot – Feature Distribution")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) > 0:
        selected_feature = st.selectbox(
            "Select a numerical feature for boxplot",
            num_cols
        )

        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_feature], ax=ax)
        ax.set_title(f"Boxplot of {selected_feature}")
        st.pyplot(fig)
    else:
        st.warning("No numerical columns found for boxplot.")

    # ------------------ Correlation Heatmap ------------------
    st.subheader("Correlation Heatmap")

    if len(num_cols) > 1:
        corr = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.warning("Not enough numerical columns for correlation heatmap.")

    # ------------------ Logistic Regression Model ------------------
    st.subheader("Logistic Regression Model")

    # Features & target (assumes last column is target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ------------------ Model Performance ------------------
    st.subheader("Model Performance")

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    st.write("ROC-AUC Score:", roc_auc)

    # ------------------ ROC Curve ------------------
    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    st.pyplot(fig)
