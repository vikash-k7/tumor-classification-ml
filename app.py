"""Interactive Streamlit demo for the binary classification model."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from src.ml_pipeline import train_and_evaluate, TARGET_NAMES

st.set_page_config(page_title="Tumor Classification ML Demo",
                   page_icon="🧪", layout="wide")

@st.cache_resource
def get_results():
    # Train once and cache for the lifetime of the Streamlit session.
    return train_and_evaluate()

result = get_results()
model = result["model"]
dataset = result["dataset"]
feature_names = list(result["feature_names"])

st.title("🧪 Tumor Classification")
st.caption("An educational binary-classification demo using scikit-learn.")
st.warning(
    "Educational use only — this model is not a medical device and must not "
    "be used to diagnose, screen, or guide treatment."
)

tab_predict, tab_performance, tab_about = st.tabs(
    ["Make a prediction", "Model performance", "About the project"]
)

with tab_predict:
    st.subheader("Enter measurement values")
    st.write(
        "Choose a sample profile to populate the fields, then adjust values "
        "to explore how the model responds."
    )

    sample_index = st.selectbox(
        "Starting sample",
        options=list(range(min(20, len(dataset.data)))),
        format_func=lambda i: f"Sample {i + 1}",
    )
    sample = dataset.data[sample_index]
    values = []

    with st.form("prediction_form"):
        columns = st.columns(3)
        for i, feature in enumerate(feature_names):
            with columns[i % 3]:
                value = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=0.0,
                    value=float(sample[i]),
                    format="%.5f",
                    key=f"feature_{i}",
                )
                values.append(value)
        submitted = st.form_submit_button("Predict class", type="primary")

    if submitted:
        input_frame = pd.DataFrame([values], columns=feature_names)
        prediction = int(model.predict(input_frame)[0])
        probabilities = model.predict_proba(input_frame)[0]
        label = TARGET_NAMES[prediction]
        confidence = float(probabilities[prediction]) * 100

        st.divider()
        if label == "benign":
            st.success(f"Model prediction: {label.title()}")
        else:
            st.error(f"Model prediction: {label.title()}")
        st.metric("Model confidence for predicted class", f"{confidence:.1f}%")
        st.caption(
            "This is a model output, not a clinical conclusion. "
            "A qualified healthcare professional must interpret medical findings."
        )

with tab_performance:
    st.subheader("Held-out test set performance")
    metrics = result["metrics"]
    metric_cols = st.columns(5)
    for col, (key, label) in zip(
        metric_cols,
        [("accuracy", "Accuracy"), ("precision", "Precision"),
         ("recall", "Recall"), ("f1_score", "F1 score"),
         ("roc_auc", "ROC-AUC")]
    ):
        col.metric(label, f"{metrics[key]:.3f}")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            result["confusion_matrix"], display_labels=TARGET_NAMES
        ).plot(ax=ax, colorbar=False)
        ax.set_title("Confusion matrix")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(
            result["y_test"], result["probabilities"], ax=ax
        )
        ax.set_title("ROC curve")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Full classification report"):
        st.code(result["report"])

with tab_about:
    st.subheader("Project overview")
    st.markdown(
        """
        **Problem:** binary classification of tumor samples into benign and
        malignant classes.

        **Dataset:** Wisconsin Diagnostic Breast Cancer dataset bundled with
        scikit-learn (569 samples, 30 numeric features).

        **Workflow:** stratified train/test split → feature standardization →
        Logistic Regression → evaluation.

        **Why a pipeline?** The scaler is fitted only on training data and the
        same transformation is applied during prediction, helping prevent
        data leakage.
        """
    )
    st.write("Training samples:", metrics["train_samples"])
    st.write("Test samples:", metrics["test_samples"])
    st.write("Input features:", metrics["features"])
