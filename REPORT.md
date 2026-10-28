# Multi-Disease Risk Prediction Using Classical Machine Learning

## 1. Abstract

This project presents a multi-disease screening system for three clinically important conditions: chronic kidney disease (CKD), diabetes, and heart disease. The system is implemented as a machine learning pipeline that trains, compares, and selects classical models for each disease using public tabular datasets. The final workflow saves the best-performing model per disease and exposes it through a Streamlit-based application that accepts patient clinical values and returns a risk or no-risk decision with a probability estimate.

The proposed system emphasizes reproducibility, leakage-safe preprocessing, and simplicity. All preprocessing steps are performed inside scikit-learn pipelines so they are fitted only on training data. The models used are logistic regression, K-nearest neighbours, random forest, and a soft voting ensemble. The app is designed as a screening aid, not a medical diagnosis tool, and clearly displays a disclaimer to reflect this limitation.

The study demonstrates that disease-specific models can be trained and evaluated in a lightweight, explainable structure suitable for academic use and early screening workflows. The resulting system is reproducible, transparent, and easy to extend to other diseases or larger clinical datasets.

---

## 2. Introduction

Early detection is essential for chronic diseases because treatment works best when symptoms are identified before complications become severe. Chronic kidney disease, diabetes, and heart disease are all major contributors to morbidity and mortality worldwide. In clinical settings, screening systems can support decision-making by giving doctors a quick risk estimate to guide further testing or follow-up.

Machine learning offers a practical way to build such screening tools, especially when the data are tabular and clinical in nature. Classical machine learning models are often preferred in academic and small-scale clinical projects because they are transparent, explainable, and robust on small datasets. This project focuses on that middle ground: a compact but realistic ML workflow that is easy to inspect, retrain, and deploy in a lightweight app.

The goal of this project is not to claim diagnosis-level accuracy, but to create a screening prototype that supports early risk assessment with clear limitations. The app allows a user to select a disease, input values, and receive a binary classification result along with a probability score.

---

## 3. Problem Statement

Medical datasets usually contain a mix of numeric and categorical clinical features, missing values, and class imbalance. In addition, screening systems must avoid false reassurance because missed positive cases can be clinically serious. A screening system should therefore be trained carefully, evaluated honestly, and deployed with safeguards.

This project addresses the following questions:

- Can classical machine learning models predict CKD, diabetes, and heart disease risk from public tabular datasets?
- Can the preprocessing be made leakage-safe so performance estimates are valid?
- Can the system select the best model per disease using a transparent metric like F1 score?
- Can the trained models be saved and reused in a simple interactive application?

---

## 4. Objectives

The key objectives of the project are:

1. To collect and verify public clinical datasets for CKD, diabetes, and heart disease.
2. To clean and prepare the data using disease-specific missing-value rules.
3. To build leakage-safe preprocessing pipelines for each model.
4. To compare four classical ML models using cross-validation.
5. To select the best model for each disease using mean F1 score.
6. To evaluate the selected model on the held-out test set.
7. To save the best model and metadata for each disease.
8. To build a Streamlit app for risk prediction.
9. To produce a clear and reusable project structure for academic evaluation.

---

## 5. System Architecture

The system can be understood as a modular pipeline with three major layers:

### 5.1 Data layer

This layer handles the raw datasets stored under the `data/` directory.

- CKD dataset saved as `data/ckd_raw.csv`
- Diabetes dataset saved as `data/diabetes.csv`
- Heart disease dataset saved as `data/heart.csv`

The data layer also stores the dataset documentation file: `data/README_DATA.md`.

### 5.2 Model training layer

This layer consists of the training and evaluation modules in the `src/` folder.

- `config.py` defines constants such as `RANDOM_STATE`, `N_FOLDS`, `TEST_SIZE`, and dataset metadata.
- `data_loader.py` cleans the data and returns `X, y`.
- `pipeline.py` builds a preprocessing and classification pipeline.
- `models.py` constructs the four required models.
- `train.py` performs the train/test split, cross-validation, model selection, test evaluation, and saving steps.
- `evaluate.py` generates confusion matrices and comparison plots.
- `verify_data.py` prints dataset shape, class distribution, and missing values.

This layer ensures the process is reproducible and the metrics come from the actual data.

### 5.3 Application layer

The application layer is implemented in `app/app.py`.

It does the following:

- loads the selected disease model
- reads the associated metadata JSON
- builds input widgets from the expected feature schema
- accepts clinical values from the user
- calls `predict_proba()`
- converts the probability into a label using threshold 0.5
- displays the result and a disclaimer

This makes the system useful as a demonstration and as an early prototype for screening support.

---

## 6. System Flow Diagram

The system follows this flow:

1. Load dataset
2. Mark missing values
3. Apply stratified 80/20 split
4. Fit preprocessing on training data only
5. Train each model
6. Validate via cross-validation
7. Average the validation scores
8. Choose the best model by mean F1
9. Test on the hold-out set once
10. Retrain on all data and save the model
11. User picks a disease
12. User enters clinical values
13. App loads the model and metadata
14. App predicts risk and shows probability and disclaimer

This workflow is represented in the project diagrams located in the `docs/` folder:

- `docs/flow_diagram.png`
- `docs/architecture.png`

---

## 7. Data Sources and Dataset Description

### 7.1 Chronic Kidney Disease (CKD)

The CKD dataset was obtained from the UCI Machine Learning Repository using the `ucimlrepo` package. The dataset contains medical measurements related to kidney function and disease status.

Key characteristics:

- Rows: 400
- Features: 24
- Target: `class`
- Positive class: `ckd`
- Distribution: 248 ckd / 152 notckd

The dataset contains missing values represented by `?` and requires string trimming and type conversion. Numeric columns are converted with `pd.to_numeric(errors="coerce")` where needed.

### 7.2 Diabetes

The diabetes data used the standard Pima Indians Diabetes benchmark, downloaded from the Brownlee dataset source as a fallback route permitted by the project instructions.

Key characteristics:

- Rows: 768
- Features: 8
- Target: `Outcome`
- Positive class: `1`
- Distribution: 268 diabetic / 500 not diabetic

Important missing-value rule: zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing because they are impossible physiological values.

### 7.3 Heart Disease

The heart disease dataset was obtained from the Kaggle source and stored as `data/heart.csv`.

Key characteristics:

- Rows: 918
- Features: 11
- Target: `HeartDisease`
- Positive class: `1`
- Distribution: 508 heart disease / 410 no disease

Missing-value rules include treating zero in `Cholesterol` and `RestingBP` as missing; categorical variables such as `Sex` and `ChestPainType` are retained as strings and handled by one-hot encoding inside the pipeline.

---

## 8. Preprocessing Strategy

A major requirement of the project was to avoid data leakage. To satisfy this, all preprocessing was implemented inside scikit-learn `Pipeline` objects.

### 8.1 Missing value rules

The project handles missing values using dataset-specific logic:

- Remove whitespace from string cells
- Treat missing markers like `?` as `NaN`
- Replace impossible zeros in clinical columns with `NaN`
- Convert numeric-looking object columns into real numeric columns

### 8.2 Feature engineering and transformation

For each pipeline:

- numeric columns are imputed using median
- categorical columns are imputed using most frequent value
- numeric columns are scaled using `StandardScaler`
- categorical columns are one-hot encoded using `OneHotEncoder(handle_unknown="ignore")`
- feature selection is applied using `SelectKBest(score_func=f_classif, k="all")`

This approach keeps the process consistent and modular while making sure the preprocessing is fitted only on training data.

---

## 9. Models and Why They Were Chosen

Four models are used for every disease:

### 9.1 Logistic Regression

Used as a simple, interpretable baseline. It is efficient and performs well on many tabular datasets.

### 9.2 K-Nearest Neighbours (KNN)

Used as a distance-based non-parametric model to capture local decision boundaries.

### 9.3 Random Forest

Used because it handles mixed data effectively and tends to be robust for structured medical datasets.

### 9.4 Soft Voting Ensemble

Used to combine the predictive power of the three base models. This ensemble makes decisions using averaged probabilities and is a sensible choice for a screening system.

The project intentionally avoids deep learning, stacking, and large hyperparameter search because the datasets are small and the scope is academic.

---

## 10. Evaluation Method

The evaluation follows a standard and defensible approach.

### 10.1 Train/test split

Each dataset is first split into 80% training and 20% test with `stratify=y` and `random_state=42`.

### 10.2 Cross-validation

The training subset is evaluated with 5-fold stratified cross-validation. This prevents overfitting to one split and gives a better estimate of general performance.

### 10.3 Metrics

The following metrics are computed:

- accuracy
- precision
- recall
- F1 score

The primary model selection metric is mean F1 score because it balances precision and recall, which is especially important in medical screening where false negatives are costly.

### 10.4 Recall check

The project also checks whether another model has recall higher by more than 0.03 while staying within 0.02 F1 of the best model. This is important because screening prioritizes finding true disease cases.

---

## 11. Results Summary

The final model selection results are as follows:

| Disease | Best model | Test accuracy | Test precision | Test recall | Test F1 |
|---|---|---:|---:|---:|---:|
| CKD | Random Forest | 0.9750 | 0.9615 | 1.0000 | 0.9804 |
| Diabetes | Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 |
| Heart disease | Soft Voting Ensemble | 0.9076 | 0.9126 | 0.9216 | 0.9171 |

### Interpretation

- CKD achieved the strongest performance, indicating that the selected clinical features are highly informative for kidney disease screening.
- Diabetes was the hardest task among the three, reflecting the difficulty of separating the classes on this dataset.
- Heart disease performed well, with a strong balance of sensitivity and precision.

These scores were generated from actual data and saved in the `results/` folder as CSV outputs and visual plots.

---

## 12. Results Files Generated

The project saves artifacts in dedicated directories:

### `results/`

- `cv_results_ckd.csv`
- `cv_results_diabetes.csv`
- `cv_results_heart.csv`
- `test_results.csv`
- `confusion_ckd.png`
- `confusion_diabetes.png`
- `confusion_heart.png`
- `comparison.png`
- `data_verification.txt`

### `models/`

- `ckd_model.pkl`
- `diabetes_model.pkl`
- `heart_model.pkl`
- `ckd_meta.json`
- `diabetes_meta.json`
- `heart_meta.json`

These files are the working outputs of the model-building pipeline and are used by theapp to make predictions.

---

## 13. Streamlit App Design

The application is implemented in `app/app.py` and provides:

- a disease selector
- numerical inputs for numeric features
- categorical dropdowns for categorical features
- a predict button
- probability output
- class label (Risk / No Risk)
- a disclaimer that the system is not a diagnosis
- an About expander that shows dataset and score information

The design is minimal and intentionally easy to use for demonstration.

---

## 14. Practical Significance

The project presents a realistic, small-scale workflow for disease screening using classical ML. It demonstrates that a simple but rigorous pipeline can achieve useful results without requiring complex infrastructure or deep learning systems.

This makes it suitable for:

- academic learning
- student mini-project demonstration
- early disease screening prototypes
- extension to other diseases and datasets

---

## 15. Limitations and Ethical Considerations

This project is a screening prototype and should not be treated as a clinical diagnosis system.

Important limitations include:

- data are from public benchmark datasets rather than full clinical studies
- the project is not externally validated on new real-world patient data
- the models may not generalize across populations
- the app is only a demonstration tool for academic use

Because of these limitations, the app includes a disclaimer stating that it is a screening aid and not a medical diagnosis.

---

## 16. Conclusion

This project successfully demonstrates the end-to-end development of a multi-disease screening system using classical machine learning. It trains and compares models for CKD, diabetes, and heart disease, applies leakage-safe preprocessing, selects the best model per disease, saves the models, and exposes them through a Streamlit interface.

The system is transparent, simple to understand, and appropriate for coursework or early healthcare AI experimentation. It shows how a healthcare screening system can be built in a controlled and explainable way without relying on complex black-box models.

---

## 17. Appendix: Project Workflow Summary

The complete academic workflow is:

1. Download public clinical datasets.
2. Verify dataset shape and missingness.
3. Clean and preprocess features safely.
4. Split into train and test sets.
5. Run cross-validation across four classical models.
6. Choose the best model by mean F1.
7. Evaluate once on the held-out test set.
8. Save the model and data schema metadata.
9. Load the model in a Streamlit app.
10. Predict risk for the user-selected disease.
11. Display results and a screening disclaimer.

This end-to-end flow is a good example of how small-scale ML systems can support early medical screening in a transparent and interpretable way.
