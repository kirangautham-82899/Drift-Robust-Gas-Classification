"""The classifiers used in the project.

Settings are the usual library defaults, fixed in advance. They are NOT tuned
here; tuning on training batches only is done in Step 8.
"""
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import SEED, make_pipeline

BASELINE_NAMES = ["kNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes"]
BOOSTING_NAMES = ["AdaBoost", "Gradient Boosting"]

PCA_VARIANCE = 0.95   # PCA keeps as many components as needed for 95% of the training variance
LDA_COMPONENTS = 4    # 5 gases -> at most 4 LDA components

# dimensionality reduction options: name -> function that builds a fresh reducer (or None)
REDUCTIONS = {
    "None": lambda: None,
    "PCA": lambda: PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED),
    "LDA": lambda: LinearDiscriminantAnalysis(n_components=LDA_COMPONENTS),
}


def get_boosting(reduction="None"):
    """Return {name: builder} for the two boosting models (same pipeline layout as the baselines).

    AdaBoost uses its default weak learner (a depth-1 tree) with 100 rounds.
    Gradient Boosting uses 100 depth-3 trees with learning rate 0.1.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {list(REDUCTIONS)}")
    reducer = REDUCTIONS[reduction]
    return {
        "AdaBoost": lambda: make_pipeline(
            AdaBoostClassifier(n_estimators=100, random_state=SEED), reducer()),
        "Gradient Boosting": lambda: make_pipeline(
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                       random_state=SEED), reducer()),
    }


def get_baselines(reduction="None"):
    """Return {name: function that builds a fresh scaler [+ reducer] + classifier pipeline}."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {list(REDUCTIONS)}")
    reducer = REDUCTIONS[reduction]
    return {
        "kNN": lambda: make_pipeline(KNeighborsClassifier(n_neighbors=5), reducer()),
        "SVM": lambda: make_pipeline(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED), reducer()),
        "Decision Tree": lambda: make_pipeline(DecisionTreeClassifier(random_state=SEED), reducer()),
        "Random Forest": lambda: make_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), reducer()),
        "Naive Bayes": lambda: make_pipeline(GaussianNB(), reducer()),
    }
