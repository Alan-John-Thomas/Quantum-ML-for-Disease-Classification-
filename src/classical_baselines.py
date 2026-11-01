"""
Classical Baseline Classifiers module.
Provides Logistic Regression, Linear SVM, RBF SVM, and Random Forest models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_classical_baselines(random_state: int = 42) -> dict:
    """
    Returns a dictionary of standardized classical baseline classifiers.
    """
    return {
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "Classical SVM (Linear)": SVC(kernel="linear", random_state=random_state),
        "Classical SVM (RBF)": SVC(kernel="rbf", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state)
    }
