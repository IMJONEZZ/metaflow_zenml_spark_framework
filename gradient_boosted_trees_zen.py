# Gradient Boosted Trees pipeline using ZenML
"""
This file implements the same functionality as ``gradient_boosted_trees_flow.py``
but uses ZenML instead of Metaflow.
It loads the Iris dataset, trains an XGBoost classifier with cross‑validation,
and prints the mean accuracy and standard deviation.
"""

from typing import Tuple, List
import numpy as np

# ZenML imports
from zenml import pipeline, step

# Hyper‑parameters (mirroring the Metaflow defaults)
RANDOM_STATE = 12
N_ESTIMATORS = 10
EVAL_METRIC = "mlogloss"
K_FOLD = 5

@step
def start() -> Tuple[List[List[float]], List[int]]:
    """Load the Iris dataset and return features/labels as JSON‑serializable lists."""
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris["data"].tolist()   # list of lists (floats)
    y = iris["target"].tolist()  # list of ints
    return X, y

@step
def train_xgb(X: List[List[float]], y: List[int]) -> List[float]:
    """Train an ``XGBClassifier`` and return the cross‑validation scores as a JSON‑serializable list.

    The parameters are identical to those used in the Metaflow example.
    """
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    X_arr = np.array(X)
    y_arr = np.array(y)

    clf = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        eval_metric=EVAL_METRIC,
        use_label_encoder=False,
    )
    scores = cross_val_score(clf, X_arr, y_arr, cv=K_FOLD)
    return scores.tolist()

@step
def end(scores: List[float]) -> None:
    """Print the mean accuracy and standard deviation.

    The output format matches the original Metaflow flow.
    """
    import numpy as np
    mean = round(100 * float(np.mean(scores)), 3)
    std = round(100 * float(np.std(scores)), 3)
    print(f"Gradient Boosted Trees Model Accuracy: {mean} \u00B1 {std}%")

@pipeline
def gradient_boosted_trees_pipeline():
    X, y = start()
    scores = train_xgb(X=X, y=y)
    end(scores=scores)

if __name__ == "__main__":
    # Running the pipeline locally via ZenML's default orchestrator.
    gradient_boosted_trees_pipeline()
