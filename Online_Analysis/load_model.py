import os, pickle
from typing import Any, Dict, Optional, Tuple


def load_trained_model(
    model_path: str,
) -> Tuple[Any, float, Optional[Any], Optional[list], Dict]:
    """
    Returns:
      model         -> Pipeline or XGBClassifier
      threshold     -> float (defaults to 0.50 if not saved)
      scaler        -> legacy compat (None if not used)
      train_channels-> list or None
      feature_meta  -> dict (fs, features_per_channel, etc.)
    Supports:
      - New format: {"model": ..., "threshold": ..., "scaler": ..., "feature_meta": {...}}
      - Legacy format: tuple/list like (model, scaler, channels) or dict missing fields
    """
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    # New format (recommended)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        threshold = float(obj.get("threshold", 0.50))
        scaler = obj.get("scaler", None)
        meta = obj.get("feature_meta", {}) or {}
        channels = meta.get("channels", None)
        return model, threshold, scaler, channels, meta

    # Legacy tuple/list: (model, scaler, channels)
    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        model = obj[0]
        scaler = obj[1] if len(obj) > 1 else None
        channels = obj[2] if len(obj) > 2 else None
        meta = {}
        threshold = 0.50
        return model, threshold, scaler, channels, meta

    # Fallback
    return obj, 0.50, None, None, {}


def get_n_features_in(model) -> Optional[int]:
    # Pipeline → last step’s attribute
    try:
        if hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf", None)
            if clf is not None and hasattr(clf, "n_features_in_"):
                return int(clf.n_features_in_)
        # Plain estimator
        if hasattr(model, "n_features_in_"):
            return int(model.n_features_in_)
    except Exception:
        pass
    return None
