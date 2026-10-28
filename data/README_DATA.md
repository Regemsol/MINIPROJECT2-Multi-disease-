# Data sources

This project uses three datasets. Raw CSV files are intentionally kept out of version control and are downloaded into the data/ folder.

## CKD
- Source: UCI Chronic Kidney Disease dataset
- URL: https://archive.ics.uci.edu/dataset/336
- Download method: `pip install ucimlrepo` and then `from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=336)`.
- Output: combined features and target saved to `data/ckd_raw.csv`.

## Diabetes
- Source: Pima Indians Diabetes dataset (NIDDK / UCI-style benchmark)
- Download method: fallback source used here: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
- Columns: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`
- Output: saved to `data/diabetes.csv` with the header above.

## Heart
- Source: Heart Failure Prediction dataset by fedesoriano
- URL: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- Download method: file placed in `data/heart.csv` after download.
- Output: `data/heart.csv`
