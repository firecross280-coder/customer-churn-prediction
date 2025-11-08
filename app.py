"""Streamlit app for training and using the churn model.

Run with:
    streamlit run app.py
"""
import io
import os
import pandas as pd
import streamlit as st

from src.model_utils import demo_synthetic, train_model_from_df, save_model, load_model, predict_df, list_saved_models, delete_saved_model
from src import tuner

st.set_page_config(page_title="Customer Churn Trainer & Inference", layout="wide")

st.title("Customer Churn — Train & Inference")

with st.sidebar:
    st.header("Data source")
    data_option = st.radio("Use data:", ("Synthetic demo", "Upload CSV"))

    uploaded_file = None
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.markdown("---")
    st.header("Model")
    target_column = st.text_input("Target column name", value="churn")
    test_size = st.slider("Test size fraction", min_value=0.05, max_value=0.5, value=0.2)
    st.markdown("---")
    st.header("Feature mapping & preprocessing")
    allow_manual_mapping = st.checkbox("Manually map column types (numeric/categorical)", value=False)

    st.markdown("---")
    st.header("Hyperparameter tuning")
    enable_tuning = st.checkbox("Enable hyperparameter tuning (RandomizedSearch)", value=False)
    n_iter = st.number_input("Tuning iterations (n_iter)", min_value=5, max_value=200, value=20)
    cv = st.number_input("CV folds", min_value=2, max_value=10, value=3)

    st.markdown("---")
    st.write("Saved models")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    models = sorted([f for f in os.listdir(models_dir) if f.endswith('.pkl')], reverse=True)
    selected_model = st.selectbox("Select saved model (for inference)", options=[None] + models)

# Load data
if data_option == "Synthetic demo":
    df = demo_synthetic(1000)
    st.info("Using synthetic dataset. To use your own data, choose 'Upload CSV' in the sidebar.")
else:
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} cols)")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head())

    if target_column not in df.columns:
        st.warning(f"Target column '{target_column}' not found in dataframe. Please set the correct target column in the sidebar.")

    # Manual feature mapping UI
    numeric_cols = []
    categorical_cols = []
    if allow_manual_mapping:
        st.subheader("Map each column type")
        with st.form(key='mapping'):
            for col in [c for c in df.columns if c != target_column]:
                typ = st.selectbox(f"Type for {col}", options=["auto", "numeric", "categorical"], key=f"map_{col}")
                if typ == "numeric":
                    numeric_cols.append(col)
                elif typ == "categorical":
                    categorical_cols.append(col)
            st.form_submit_button("Save mapping")

    if not allow_manual_mapping:
        # Infer simple mapping
        numeric_cols = [c for c in df.columns if c != target_column and pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c not in numeric_cols and c != target_column]

    st.markdown("---")
    if st.button("Train model"):
        try:
            if enable_tuning:
                # Simple param distributions; expose a few common RF params
                param_distributions = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                }
                search = tuner.tune_pipeline(df, target_column, numeric_cols, categorical_cols,
                                             param_distributions=param_distributions, n_iter=int(n_iter), cv=int(cv))
                pipeline = search.best_estimator_
                metrics = {
                    'best_params': search.best_params_,
                    'best_score': float(search.best_score_),
                }
                st.success("Tuning completed")
                st.subheader("Tuning results")
                st.json(metrics)
            else:
                pipeline, metrics, X_test, y_test = train_model_from_df(df, target=target_column, test_size=test_size)
                st.success("Training completed")

                st.subheader("Metrics")
                st.json(metrics)

            path = save_model(pipeline)
            st.info(f"Saved model to {path}")

            # Show sample predictions when not tuning (we have X_test)
            try:
                sample = X_test.head(5)
                preds = predict_df(pipeline, sample)
                st.subheader("Sample predictions")
                st.dataframe(preds)
            except Exception:
                pass
        except Exception as e:
            st.error(f"Training failed: {e}")

# Inference using a selected saved model
st.markdown("---")
st.header("Inference")
if selected_model:
    model_path = os.path.join(models_dir, selected_model)
    try:
        pipeline = load_model(model_path)
        st.success(f"Loaded model: {selected_model}")

        st.subheader("Batch predict using CSV")
        inf_file = st.file_uploader("Upload CSV for prediction (features only)", type=["csv"], key="inf")
        if inf_file is not None:
            try:
                inf_df = pd.read_csv(inf_file)
                res = predict_df(pipeline, inf_df)
                st.dataframe(res.head(20))

                # Provide download
                towrite = io.BytesIO()
                res.to_csv(towrite, index=False)
                towrite.seek(0)
                st.download_button("Download predictions CSV", towrite, file_name="predictions.csv")
            except Exception as e:
                st.error(f"Failed to run predictions: {e}")

        st.subheader("Quick predict using example row")
        st.write("If the model expects the original training features, provide a single-row CSV or use synthetic demo to get sample inputs.")
        if st.button("Predict on synthetic sample"):
            demo = demo_synthetic(1).drop(columns=[c for c in demo_synthetic(1).columns if c == target_column], errors='ignore')
            result = predict_df(pipeline, demo)
            st.dataframe(result)

    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.info("No saved model selected. Train one above or upload a model to the models/ directory.")

st.markdown("---")
st.header("Model registry")
saved = list_saved_models(models_dir=models_dir)
if saved:
    st.write("Registered models (index)")
    for m in saved:
        cols = st.columns([3, 2, 2, 1])
        cols[0].write(m.get('filename'))
        cols[1].write(m.get('saved_at'))
        cols[2].write(m.get('n_features'))
        if cols[3].button(f"Delete##{m.get('filename')}"):
            ok = delete_saved_model(m.get('filename'), models_dir=models_dir)
            if ok:
                st.success(f"Deleted {m.get('filename')}")
            else:
                st.error(f"Failed to delete {m.get('filename')}")
            st.experimental_rerun()
else:
    st.info("No registered models in index.json yet.")
