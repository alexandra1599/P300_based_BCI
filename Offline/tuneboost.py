"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, recall_score
import numpy as np
import warnings


def tune_xgboost(X_train, y_train, cv=5, scoring="f1_weighted", verbose=1):
    """
    Perform grid search hyperparameter tuning for XGBoost.

    Parameters:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        cv (int): Number of cross-validation folds.
        scoring (str or callable): Scoring function.
        verbose (int): Verbosity level.

    Returns:
        best_model (XGBClassifier): Best XGBoost model.
        best_params (dict): Best parameter combination.
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "gamma": [0, 1],
    }

    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print("✅ Best Parameters Found:")
    print(grid_search.best_params_)

    print("\n📊 Classification Report on Training Set:")
    y_pred = grid_search.best_estimator_.predict(X_train)
    print(classification_report(y_train, y_pred))

    return grid_search.best_estimator_, grid_search.best_params_
