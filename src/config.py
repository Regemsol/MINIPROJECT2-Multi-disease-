RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.20
KNN_K = 5
RF_TREES = 300
LOGREG_MAX_ITER = 1000

DATASETS = {
    "ckd": dict(path="data/ckd_raw.csv", target="class", positive="ckd",
                zero_missing=[], drop=["id"], label="Chronic Kidney Disease"),
    "diabetes": dict(path="data/diabetes.csv", target="Outcome", positive=1,
                     zero_missing=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
                     drop=[], label="Diabetes"),
    "heart": dict(path="data/heart.csv", target="HeartDisease", positive=1,
                  zero_missing=["Cholesterol", "RestingBP"], drop=[], label="Heart Disease"),
}
