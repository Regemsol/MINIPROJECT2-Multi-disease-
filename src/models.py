import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.config import KNN_K, LOGREG_MAX_ITER, RF_TREES, RANDOM_STATE


def get_models():
    lr = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RANDOM_STATE)
    knn = KNeighborsClassifier(n_neighbors=KNN_K)
    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=RANDOM_STATE)
    ensemble = VotingClassifier(estimators=[
        ("logreg", lr),
        ("knn", knn),
        ("rf", rf),
    ], voting="soft")

    return {
        "Logistic Regression": lr,
        "KNN (k=5)": knn,
        "Random Forest": rf,
        "Soft Voting Ensemble": ensemble,
    }
