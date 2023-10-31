import os
import numpy as np
import json
from pathlib import Path

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from cancel_model.config.core import DATASET_DIR


def feature_selection(X_train: None, Y_train: None):
    feat_path = Path(f"{DATASET_DIR}/feats.json")
    if os.path.exists(feat_path):
        # load selected features
        with open(feat_path, "r", encoding="utf-8") as feat_fn:
            cols = json.load(feat_fn)
    else:
        # Recursive Feature Elimination - recursively choose best performing features
        logreg = LogisticRegression(max_iter=5000)

        rfe = RFE(logreg, n_features_to_select=10)
        rfe = rfe.fit(X_train, Y_train.values.ravel())
        col_idx = np.where(rfe.ranking_ == 1)[0].tolist()
        cols = [X_train.columns[i] for i in col_idx]

        # dump selected features to datasets
        with open(feat_path, "w", encoding="utf-8") as feat_fn:
            json.dump(cols, feat_fn)

    return cols

