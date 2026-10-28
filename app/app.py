import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_DIR = Path("models")


def load_disease_data(disease):
    model_path = MODEL_DIR / f"{disease}_model.pkl"
    meta_path = MODEL_DIR / f"{disease}_meta.json"
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def main():
    st.title("Multi-Disease Risk Prediction")
    disease = st.sidebar.selectbox("Choose disease", ["ckd", "diabetes", "heart"])
    disease_label = {
        "ckd": "Chronic Kidney Disease",
        "diabetes": "Diabetes",
        "heart": "Heart Disease",
    }[disease]

    model, meta = load_disease_data(disease)
    columns = meta["expected_input_columns"]
    row = {}

    for col in columns:
        col_type = meta["column_types"][col]
        if col_type == "numeric":
            stats = meta["numeric_summary"].get(col, {})
            value = stats.get("median", 0.0)
            min_v = stats.get("min")
            max_v = stats.get("max")
            step = 0.01 if isinstance(value, float) else 1.0
            row[col] = st.number_input(
                col,
                value=float(value),
                min_value=float(min_v) if min_v is not None else None,
                max_value=float(max_v) if max_v is not None else None,
                step=step,
            )
        else:
            choices = meta["categorical_values"].get(col, [])
            row[col] = st.selectbox(col, options=choices)

    if st.button("Predict"):
        input_df = pd.DataFrame([row], columns=columns)
        proba = model.predict_proba(input_df)[0]
        positive_index = 1 if len(proba) > 1 else 0
        risk_prob = float(proba[positive_index])
        prediction = "Risk" if risk_prob >= 0.5 else "No Risk"
        st.subheader(f"Prediction: {prediction}")
        st.write(f"Probability of disease: {risk_prob * 100:.1f}%")
        st.write(f"Model used: {meta['winning_model_name']}")
        st.write(f"Test F1: {meta['test_scores']['f1']:.4f}")
        st.caption("This is a screening aid built for an academic project. It is not a medical diagnosis. Please consult a doctor.")

    with st.expander("About"):
        st.write(f"Dataset: {disease_label}")
        st.write(f"Rows used in dataset: {meta['dataset_rows']}")
        st.write(f"Test scores: {meta['test_scores']}")


if __name__ == "__main__":
    main()
