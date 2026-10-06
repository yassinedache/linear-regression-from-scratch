"""California Housing: loading, feature construction, train/test split and standardisation.

The raw file (1990 census, one row per block group) comes from the classic
"housing.csv" dataset. The features are built like scikit-learn's
fetch_california_housing: averages per household, and the target in units of 100,000 $.
"""

import csv
import os

import numpy as np

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "housing.csv")


def load_california(path: str = DEFAULT_PATH):
    """Return (X, y, feature_names). Rows with a missing value are dropped (207 rows)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r["total_bedrooms"]:
                continue
            hh = float(r["households"])
            rows.append([
                float(r["median_income"]),
                float(r["housing_median_age"]),
                float(r["total_rooms"]) / hh,
                float(r["total_bedrooms"]) / hh,
                float(r["population"]),
                float(r["population"]) / hh,
                float(r["latitude"]),
                float(r["longitude"]),
                float(r["median_house_value"]) / 100_000,
            ])
    data = np.array(rows)
    return data[:, :-1], data[:, -1], list(FEATURES)


def train_test_split(X, y, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(round(test_size * len(y)))
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]


class Standardizer:
    """x -> (x - mean) / std, with mean and std learned on the training set only."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_
