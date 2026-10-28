# Multi-Disease Risk Prediction: Project Explanation

## 1. What this project is

This project builds a simple machine learning system that predicts whether a person is at risk for three diseases:

- Chronic Kidney Disease (CKD)
- Diabetes
- Heart disease

The app uses classical machine learning models rather than deep learning. It is meant as a screening aid, not a medical diagnosis.

We trained one model per disease and then saved it. The app loads the relevant model and asks the user to enter health values. Based on those values, the app predicts whether the patient is likely to be in the positive class:

- CKD = `ckd`
- Diabetes = `1` (diabetic)
- Heart disease = `1` (heart disease present)

The final output includes:

- Risk or No Risk
- Probability of disease presence
- Name of the winning model used
- Test F1 score for that disease
- A disclaimer that this is only a screening aid, not a diagnosis

---

## 2. Why this problem matters

These diseases often develop slowly. Early screening can help doctors detect risk earlier and act before the illness becomes severe.

Screening matters because:

- CKD can progress silently until the kidneys are badly damaged
- Diabetes may be unnoticed for a long time and can affect many organs
- Heart disease may be missed until symptoms become severe

This is why the model is built around early risk screening rather than final diagnosis.

---

## 3. The three diseases and what we predict

### 3.1 Chronic Kidney Disease

Target column: `class`

Positive class: `ckd`

Sample variables used by the model include:

- age
- blood pressure
- specific gravity
- albumin
- blood glucose random
- blood urea
- serum creatinine
- hemoglobin
- packed cell volume
- hypertension
- diabetes mellitus
- appetite
- pedal edema
- anemia

These input features are clinical indicators that are commonly associated with kidney function and kidney disease risk.

### 3.2 Diabetes

Target column: `Outcome`

Positive class: `1`

Main predictive factors used are:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

These values come from the Pima Indians Diabetes dataset and represent the main medical features used in diabetes screening models.

### 3.3 Heart Disease

Target column: `HeartDisease`

Positive class: `1`

Main predictive factors used are:

- Age
- Sex
- ChestPainType
- RestingBP
- Cholesterol
- FastingBS
- RestingECG
- MaxHR
- ExerciseAngina
- Oldpeak
- ST_Slope

These are common clinical indicators used to assess heart-risk patterns in routine diagnosis tables.

---

## 4. Machine learning workflow

The project follows a standard medical ML pipeline.

### Step 1: Load data

Each dataset is loaded from its CSV file in the `data/` folder.

### Step 2: Clean missing values

We handle missing values using rules specific to each disease:

- `?` is treated as missing
- empty cells are treated as missing
- impossible zero values are converted to missing for certain medical columns
- string values are trimmed to remove stray spaces or tabs

Examples:

- In CKD, `?` values are considered missing
- In diabetes, zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing because those values are biologically impossible
- In heart disease, zero in `Cholesterol` and `RestingBP` is treated as missing

### Step 3: Split into train and test

The dataset is split once into:

- 80% training
- 20% testing

This split uses stratification, which means class balance is preserved. The random seed is fixed at `42` for reproducibility.

### Step 4: Build preprocessing pipeline

Each model uses a scikit-learn pipeline that contains:

- Numeric imputation using median
- Categorical imputation using most frequent value
- Standard scaling for numeric columns
- One-hot encoding for categorical columns
- Feature selection with `SelectKBest`
- Classifier at the end

This is important because it ensures we do not leak information from the test data into training.

### Step 5: Cross-validation

We use 5-fold cross-validation with stratification.

This means the model is trained on several different train/validation splits, and we average the scores to estimate how well it generalizes.

The metrics used are:

- Accuracy
- Precision
- Recall
- F1 Score

### Step 6: Select best model

The best model is selected using mean F1 score.

Why F1?

- For screening problems, missing a true disease case is costly
- Accuracy alone can be misleading if the disease is imbalanced
- F1 balances precision and recall better than accuracy

### Step 7: Evaluate once on the test set

After selecting the winning model, we evaluate it once on the held-out 20% test set. This gives a realistic estimate of real-world performance.

### Step 8: Save the trained model

The winning pipeline is saved with `joblib` so the Streamlit app can use it later.

---

## 5. Models used

We compare exactly four models for each disease.

### 1. Logistic Regression

- Fast and interpretable
- Good baseline model
- Works well with many tabular datasets

### 2. KNN (k=5)

- Uses the 5 nearest samples to decide the class
- Good for understanding local patterns in the data

### 3. Random Forest

- Ensemble of decision trees
- Works well with mixed numeric and categorical data
- Often robust for healthcare-style tabular data

### 4. Soft Voting Ensemble

- Combines Logistic Regression + KNN + Random Forest
- Uses soft voting to average predicted probabilities
- Often performs well in screening tasks

---

## 6. Why we did not use deep learning

We deliberately kept the project simple because:

- the datasets are small
- the task is tabular and classical
- interpretability matters
- the instructions explicitly forbid deep learning and heavy model stacking
- the goal is a student mini project, not a production clinical engine

---

## 7. Data leakage prevention

This is a very important concept in this project.

Data leakage happens when the preprocessing steps are fitted on the entire dataset instead of only on training data. That creates unrealistic performance results.

To avoid this:

- we split the data first
- then we fit preprocessing, scaling, encoding, and feature selection inside the pipeline
- the pipeline is trained only on the training fold/partition

This makes the evaluation realistic and honest.

---

## 8. Important prediction factors in the app

The app asks for the exact input features that were used during training for that disease. These are not arbitrary values; they are the original disease variables from the dataset.

### CKD input features

- age
- bp
- sg
- al
- su
- bgr
- bu
- sc
- sod
- pot
- hemo
- pcv
- wbcc
- rbcc
- htn
- dm
- cad
- appet
- pe
- ane
- plus related categorical clinical fields based on the saved metadata

### Diabetes input features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Heart disease input features

- Age
- Sex
- ChestPainType
- RestingBP
- Cholesterol
- FastingBS
- RestingECG
- MaxHR
- ExerciseAngina
- Oldpeak
- ST_Slope

These values are taken from the patient/clinical data and fed into the model as a one-row DataFrame.

---

## 9. How the app works

When the user opens the app:

1. They choose one disease from the sidebar
2. The app loads the matching trained model file and metadata file
3. It reads the feature names and feature types from metadata
4. It builds one input row from the user-entered values
5. It runs `predict_proba()`
6. It computes the disease probability
7. It compares it with the threshold 0.5
8. It prints:
   - Risk or No Risk
   - probability percentage
   - model name
   - test F1 score
   - disclaimer

---

## 10. Why the threshold is 0.5

For binary classification, the default class decision threshold is usually 0.5.

- If probability of disease >= 0.5 => Risk
- Else => No Risk

This is simple and standard for screening models in an academic project.

---

## 11. Project folder structure

```text
multi-disease-risk-prediction/
├── app/
│   └── app.py
├── data/
│   ├── README_DATA.md
│   ├── ckd_raw.csv
│   ├── diabetes.csv
│   └── heart.csv
├── docs/
│   ├── architecture.png
│   └── flow_diagram.png
├── models/
│   ├── ckd_model.pkl
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── ckd_meta.json
│   ├── diabetes_meta.json
│   └── heart_meta.json
├── results/
│   ├── cv_results_ckd.csv
│   ├── cv_results_diabetes.csv
│   ├── cv_results_heart.csv
│   ├── test_results.csv
│   ├── confusion_ckd.png
│   ├── confusion_diabetes.png
│   ├── confusion_heart.png
│   └── comparison.png
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── pipeline.py
│   ├── models.py
│   ├── verify_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── __init__.py
├── .gitignore
├── requirements.txt
├── README.md
├── EXPLANATION.md
└── .git
```

---

## 12. Copyable step-by-step run instructions

### 12.1 Clone and set up

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 12.2 Download datasets

```bash
python - <<'PY'
from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

# CKD
obj = fetch_ucirepo(id=336)
df = obj.data.original.copy()
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()
df.to_csv('data/ckd_raw.csv', index=False)

# Diabetes fallback
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
df2 = pd.read_csv(url, header=None)
df2.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df2.to_csv('data/diabetes.csv', index=False)

print('Downloaded CKD and Diabetes')
PY
```

Then place the Kaggle heart dataset file here:

```text
data/heart.csv
```

### 12.3 Verify the raw data

```bash
python src/verify_data.py
```

### 12.4 Train models and generate results

```bash
python src/train.py
```

This will:

- run 5-fold cross-validation
- select the best model per disease
- save model files to `models/`
- save metrics to `results/`
- generate confusion matrices and comparison chart

### 12.5 Start the app

```bash
streamlit run app/app.py --server.headless true
```

Then open the local URL shown in the terminal, normally:

```text
http://localhost:8501
```

---

## 13. How to explain the project in a presentation

If you need to present it verbally, use this structure:

### Intro

"This project predicts risk for three diseases using tabular clinical data and classical machine learning models. It is designed as a screening aid rather than a diagnosis tool."

### What the project does

"We load public datasets, clean them, split them into train and test sets, preprocess them safely inside a pipeline, compare four classical models, and save the best model for each disease."

### Why it matters

"Early screening can help reduce complications and improve treatment timing for CKD, diabetes, and heart disease."

### Why preprocessing matters

"Medical data contains missing values and noisy values, so we handle them carefully and avoid data leakage by fitting preprocessing only on training data."

### Why F1 and recall are important

"Accuracy alone can hide missed positive cases. Screening is concerned with missing actual disease cases, so F1 and recall are especially important."

### What the app does

"The user selects a disease, enters values, and the app predicts the risk level along with a probability output and disclaimer."

### Final message

"This is an academic screening prototype built for learning and early risk detection, not a clinical diagnosis tool."

---

## 14. Final summary

This project is a complete mini-project pipeline built around a simple but realistic clinical ML workflow:

- download public datasets
- clean and verify them
- train leakage-safe models
- select the best model per disease
- save the trained models
- expose them through a Streamlit app

It shows the basic end-to-end process of medical risk prediction using classical machine learning and makes it easy to understand the overall logic behind disease screening systems.
