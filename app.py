
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------
#              STREAMLIT APP CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="ML Classification Models – Assignment 2",
    layout="wide"
)

st.title("📘 Machine Learning Assignment 2 – Classification Models")
st.write("Upload test data, choose a model, and view evaluation metrics.")

# -------------------------------------------------------------
#              LOAD TRAINED MODELS
# -------------------------------------------------------------
@st.cache_resource
def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

MODELS = {
    "Logistic Regression": "model/logistic_reg.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# -------------------------------------------------------------
#              SIDEBAR OPTIONS
# -------------------------------------------------------------
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model:", list(MODELS.keys()))

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# -------------------------------------------------------------
#              PROCESS WHEN FILE UPLOADED
# -------------------------------------------------------------
if uploaded_file is not None:
    st.subheader("📄 Uploaded Test Dataset")
    test_df = pd.read_csv(uploaded_file)
    st.dataframe(test_df.head())

    # Check target column
    if "target" not in test_df.columns:
        st.error("❌ The CSV must contain a 'target' column!")
        st.stop()

    # Split X, y
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    # ---------------------------------------------------------
    #              LOAD SELECTED MODEL
    # ---------------------------------------------------------
    st.subheader(f"🔍 Using Model: **{model_name}**")

    model = load_model(MODELS[model_name])

    # Predict
    y_pred = model.predict(X_test)

    # Probability for AUC
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = np.nan

    # ---------------------------------------------------------
    #              METRICS CALCULATION
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    # ---------------------------------------------------------
    #              DISPLAY METRICS
    # ---------------------------------------------------------
    st.subheader("📊 Evaluation Metrics")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric("Accuracy", round(accuracy, 4))
    metric_col1.metric("AUC Score", round(auc, 4))

    metric_col2.metric("Precision", round(precision, 4))
    metric_col2.metric("Recall", round(recall, 4))

    metric_col3.metric("F1 Score", round(f1, 4))
    metric_col3.metric("MCC Score", round(mcc, 4))

    # ---------------------------------------------------------
    #              CONFUSION MATRIX
    # ---------------------------------------------------------
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ---------------------------------------------------------
    #              CLASSIFICATION REPORT
    # ---------------------------------------------------------
    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("📤 Please upload a CSV file from the sidebar to begin.")