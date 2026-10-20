"""The baseline classifiers (Step 5).

Settings are the usual library defaults, fixed in advance. They are NOT tuned
here; tuning on training batches only is done in Step 8.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import SEED, make_pipeline

BASELINE_NAMES = ["kNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes"]


def get_baselines():
    """Return {name: function that builds a fresh scaler + classifier pipeline}."""
    return {
        "kNN": lambda: make_pipeline(KNeighborsClassifier(n_neighbors=5)),
        "SVM": lambda: make_pipeline(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
        "Decision Tree": lambda: make_pipeline(DecisionTreeClassifier(random_state=SEED)),
        "Random Forest": lambda: make_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
        "Naive Bayes": lambda: make_pipeline(GaussianNB()),
    }
