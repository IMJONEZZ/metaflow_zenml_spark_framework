# Parallel Trees pipeline using ZenML
"""
This file mirrors ``parallel_branches_flow.py`` but uses ZenML instead of Metaflow.
It loads the Iris dataset, trains a RandomForest and an XGBoost model in parallel,
computes mean accuracy and std for each, and prints the results.
All data exchanged between steps is JSON‑serializable (lists), avoiding
ZenML's built‑in container materializer limitations.
"""

from typing import List, Tuple
import numpy as np

# ZenML imports
from zenml import pipeline, step

# Hyper‑parameters (mirroring Metaflow defaults)
MAX_DEPTH = None
RANDOM_STATE = 21
N_ESTIMATORS = 10
MIN_SAMPLES_SPLIT = 2
EVAL_METRIC = "mlogloss"
K_FOLD = 5

@step
def start() -> Tuple[List[List[float]], List[int]]:
    """Load the Iris dataset and return features/labels as JSON‑serializable lists."""
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris["data"].tolist()   # list of [float]
    y = iris["target"].tolist()  # list of int
    return X, y

@step
def train_rf(X: List[List[float]], y: List[int]) -> List[float]:
    """Train a RandomForestClassifier and return its CV scores as a list."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    X_arr = np.array(X)
    y_arr = np.array(y)
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
    )
    scores = cross_val_score(clf, X_arr, y_arr, cv=K_FOLD)
    return scores.tolist()

@step
def train_xgb(X: List[List[float]], y: List[int]) -> List[float]:
    """Train an XGBClassifier and return its CV scores as a list."""
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
def score(rf_scores: List[float], xgb_scores: List[float]) -> List[Tuple[str, float, float]]:
    """Combine the results from both models into a list of (name, mean, std)."""
    import numpy as np
    results: List[Tuple[str, float, float]] = []
    for name, scores in [("Random Forest", rf_scores), ("XGBoost", xgb_scores)]:
        mean = round(100 * float(np.mean(scores)), 3)
        std = round(100 * float(np.std(scores)), 3)
        results.append((name, mean, std))
    return results

@step
def end(results: List[Tuple[str, float, float]]) -> None:
    """Print the model accuracies."""
    for name, mean, std in results:
        print(f"{name} Model Accuracy: {mean} ± {std}%")

@pipeline
def parallel_trees_pipeline():
    X, y = start()
    rf_scores = train_rf(X=X, y=y)
    xgb_scores = train_xgb(X=X, y=y)
    results = score(rf_scores=rf_scores, xgb_scores=xgb_scores)
    end(results=results)

if __name__ == "__main__":
    # Running the pipeline locally via ZenML's default orchestrator.
    parallel_trees_pipeline()
