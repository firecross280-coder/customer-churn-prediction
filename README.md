# customer-churn-prediction

This repository contains a simple, self-contained example for training a
customer churn prediction model. It generates a synthetic dataset, trains a
scikit-learn pipeline (preprocessing + Random Forest), evaluates it, and saves
the trained model to `models/model.pkl`.

Quick start
-----------

1. Create a Python virtual environment and activate it.
2. Install dependencies:

	pip install -r requirements.txt

3. Run training:

	python3 src/train.py

This will print evaluation metrics and save the trained model to `models/model.pkl`.

Notes
-----

- The script uses a synthetic dataset (no external data needed).
- The model and pipeline are intentionally simple so you can adapt them to
  your real dataset (replace the data generation step with a data loader).

Files added
-----------

- `src/train.py` — training script that generates data, trains and saves model.
- `requirements.txt` — Python dependencies.

Feel free to request enhancements like using real data, hyperparameter tuning,
cross-validation, or exporting an inference endpoint.

Streamlit app
-------------

To run the interactive app (train on your dataset or the synthetic demo and
perform inference):

1. Install dependencies (see `requirements.txt`).
2. Run:

	streamlit run app.py

The app lets you upload a CSV, choose the target column, train a model, view
metrics, save the trained model, and run batch or quick inference using saved
models located in the `models/` directory.
# customer-churn-prediction