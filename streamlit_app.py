
import os
import tempfile
import runpy

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------#
# STREAMLIT APP CONFIGURATION
# -------------------------------#
st.set_page_config(page_title="ML Classification Models – Assignment 2", layout="wide")
st.title("📘 Machine Learning Assignment 2 – Classification Models")
st.write("Upload test data, choose a model, and view evaluation metrics.")

# -------------------------------#
# CACHE DECORATOR (newer/older Streamlit)
# -------------------------------#
try:
    cache_resource = st.cache_resource  # Streamlit >= 1.18
except AttributeError:
    cache_resource = st.cache          # Fallback for older versions

# -------------------------------#
# LOADERS
# -------------------------------#
@cache_resource(show_spinner="🔄 Converting & executing notebook...")
def load_model_from_ipynb(nb_path: str):
    """
    Convert an .ipynb to .py with nbconvert, execute it once,
    and return either `trained_model` or the result of `build_model()`.
    """
    try:
        import nbformat
        from nbconvert import PythonExporter
    except Exception as e:
        raise RuntimeError(
            "This feature requires extra packages. Please run:\n"
            "pip install nbformat nbconvert"
        ) from e

    # 1) Read notebook & export to Python
    nb = nbformat.read(nb_path, as_version=4)
    code, _ = PythonExporter().from_notebook_node(nb)

    # 2) Write to a temporary .py file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        py_path = tf.name

    # 3) Execute the generated .py in an isolated namespace
    try:
        ns = runpy.run_path(py_path)
    finally:
        # Always clean up the temp file
        try:
            os.remove(py_path)
        except Exception:
            pass

    # 4) Extract the model object
    if "trained_model" in ns:
        return ns["trained_model"]
    if "build_model" in ns and callable(ns["build_model"]):
        return ns["build_model"]()

    raise ValueError(
        "Notebook must define a `trained_model` variable or a callable `build_model()`."
    )

@cache_resource(show_spinner="📦 Loading pickled model...")
def load_pickled_model(file_path: str):
    """
    Load a model from .pkl or .joblib. Refuses non-model files (e.g., JSON/.ipynb).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".joblib":
        from joblib import load as joblib_load
        return joblib_load(file_path)
    if ext != ".pkl":
        raise ValueError(f"Unsupported model file for pickle loader: {ext}")

    with open(file_path, "rb") as f:
        # quick sanity check: not JSON/notebook text
        first = f.read(1)
        if first == b"{":
            raise ValueError(
                f"{file_path} looks like JSON/text (starts with '{{'). "
                f"Provide a .pkl or use an .ipynb with the notebook loader."
            )
        f.seek(0)
        return pickle.load(f)

def load_model_any(path: str):
    """
    Dispatch to the correct loader based on file extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ipynb":
        return load_model_from_ipynb(path)
    if ext in {".pkl", ".joblib"}:
        return load_pickled_model(path)
    raise ValueError(f"Unsupported model file type: {ext} (use .ipynb, .pkl, or .joblib)")

# -------------------------------#
# MODEL REGISTRY
# -------------------------------#
MODELS = {
    # NOTE: This entry now intentionally points to the notebook.
    "Logistic Regression (Notebook)": "model/Logistic_Regression.ipynb",
    # Keep your other artifacts as pickles or joblib files
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl",
}

# -------------------------------#
# SIDEBAR
# -------------------------------#
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model:", list(MODELS.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# -------------------------------#
# WHEN FILE UPLOADED
# -------------------------------#
if uploaded_file is not None:
    st.subheader("📄 Uploaded Test Dataset")
    test_df = pd.read_csv(uploaded_file)
    st.dataframe(test_df.head())

    # Flexible target detection (avoid confusing message vs. check)
    possible_targets = ["OUTCOME", "target", "Target"]
    target_col = next((c for c in possible_targets if c in test_df.columns), None)
    if not target_col:
        st.error(f"❌ The CSV must contain one of these columns: {possible_targets}")
        st.stop()

    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    # Load selected model (supports .ipynb/.pkl/.joblib)
    st.subheader(f"🔍 Using Model: **{model_name}**")
    model_or_artifact = load_model_any(MODELS[model_name])

    # Support both plain estimator or dict artifact {"model": ..., "features": [...]}
    if isinstance(model_or_artifact, dict) and "model" in model_or_artifact:
        model = model_or_artifact["model"]
        if "features" in model_or_artifact:
            # Align columns if the artifact contains a training schema
            X_test = X_test.reindex(columns=model_or_artifact["features"], fill_value=0)
    else:
        model = model_or_artifact

    # Predict
    y_pred = model.predict(X_test)

    # Probability / score for AUC (handles binary & multiclass; graceful fallback)
    auc = np.nan
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_score)
    except Exception:
        # Keep auc as NaN if scoring isn't applicable
        pass

    # -------------------------------#
    # METRICS
    # -------------------------------#
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("📊 Evaluation Metrics")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Accuracy", round(accuracy, 4))
    metric_col1.metric("AUC Score", round(auc, 4) if not np.isnan(auc) else "NA")
    metric_col2.metric("Precision", round(precision, 4))
    metric_col2.metric("Recall", round(recall, 4))
    metric_col3.metric("F1 Score", round(f1, 4))
    metric_col3.metric("MCC Score", round(mcc, 4))

    # -------------------------------#
    # CONFUSION MATRIX
    # -------------------------------#
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    labels = np.unique(y_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -------------------------------#
    # CLASSIFICATION REPORT
    # -------------------------------#
    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("📤 Please upload a CSV file from the sidebar to begin.")
