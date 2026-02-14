import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------------------
#                  STREAMLIT CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 - Classification Model",
    layout="wide"
)

st.title("📘 Machine Learning Assignment 2 – Insurance Claim")
st.write("Upload test data to predict the Insurance Claim, choose a model, and view evaluation metrics.")

# -------------------------------------------------------------
#                  LOAD MODELS
# -------------------------------------------------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Make sure these files exist inside model/ folder
MODELS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn_model.pkl",
    "Naive Bayes": "model/gaussian_nb_model.pkl",
    "Random Forest": "model/random_forest_model.pkl",
    "XGBoost": "model/xgboost_best_model.pkl"
}

# -------------------------------------------------------------
#                  SIDEBAR
# -------------------------------------------------------------
st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a Model:",
    list(MODELS.keys())
)

st.sidebar.markdown("---")
# Download sample test data
st.sidebar.subheader("📥 Download Sample Test Data by clicking below")

with open("test_data.csv", "rb") as file:
    st.sidebar.download_button(
        label="test_data.csv",
        data=file,
        file_name="test_data.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

st.sidebar.markdown("---")

#Execute with default test data
st.sidebar.subheader("📥 Run with Default Test Data by clicking below")
run_default = st.sidebar.button("▶ Run with Default Test Data")


# -------------------------------------------------------------
#                  MAIN LOGIC
# -------------------------------------------------------------
test_df = None

# If user uploads a file
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Using uploaded test data")

# If button clicked → use default file
elif run_default:
    test_df = pd.read_csv("test_data.csv")
    st.success("Using default test_data.csv")
    
# If dataset selected
if test_df is not None:
    st.subheader("📄 Uploaded Test Dataset")
    st.dataframe(test_df.head())

    # Check target column
    if "OUTCOME" not in test_df.columns:
        st.error("❌ The CSV must contain an 'OUTCOME' column.")
        st.stop()

    # Split features and target
    X_test = test_df.drop("OUTCOME", axis=1)
    y_test = test_df["OUTCOME"]

    # Load selected model
    model = load_model(MODELS[model_name])
    st.success(f"✅ Loaded Model: {model_name}")

    # ---------------------------------------------------------
    #                  PREDICTIONS
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)

    # AUC calculation
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = np.nan

    # ---------------------------------------------------------
    #                  METRICS
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 4))
    col1.metric("AUC Score", round(auc, 4) if not np.isnan(auc) else "N/A")

    col2.metric("Precision", round(precision, 4))
    col2.metric("Recall", round(recall, 4))

    col3.metric("F1 Score", round(f1, 4))
    col3.metric("MCC Score", round(mcc, 4))

    # ---------------------------------------------------------
    #                  CONFUSION MATRIX
    # ---------------------------------------------------------
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,0], yticklabels=[1,0], annot_kws={"size": 9})
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    st.pyplot(fig, width='content')

    # ---------------------------------------------------------
    #                  CLASSIFICATION Evaluation REPORT
    # ---------------------------------------------------------
    st.subheader("📄 Classification Evalutaion Report")

    metrics_df = pd.DataFrame(
    {
        "Model": [model_name],
        "Accuracy": [accuracy],
        "AUC": [auc],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "MCC": [mcc]
    }
    )

    st.dataframe(metrics_df)

else:
    st.info("📤 Please upload a CSV file from the sidebar to begin.")