from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import numpy as np
from typing import Optional
from enum import Enum


class Parameters(Enum):
    X = 1
    Y = 2
    Z = 3
    H = 4
    PHI = 5
    ALPHA = 6
    SOMA_RADIUS = 7
    TRUNK_LENGTH = 8
    TRUNK_RADIUS = 9
    DENDRITE_RADIUS = 10
    TUFT_RADIUS = 11
    DENDRITE_LENGTH = 12
    GMAX = 13


class ClassifierTypes(Enum):
    LINEAR_REGRESSION = 1
    RIDGE_REGRESSION = 2
    SUPPORT_VECTOR_REGRESSION = 3


class ClassifierBuilder(object):
    def __init__(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                 clf: Optional[ClassifierTypes] = None) -> None:
        self.x = x
        self.y = y
        if clf == ClassifierTypes.LINEAR_REGRESSION:
            self.clf = LinearRegression()
        elif clf == ClassifierTypes.RIDGE_REGRESSION:
            self.clf = Ridge()
        elif clf == ClassifierTypes.SUPPORT_VECTOR_REGRESSION:
            self.clf = SVR()
        else:
            self.clf = None

    def fit(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> None:
        if x and y:
            self.clf = self.clf.fit(x, y)
        elif isinstance(self.x, np.ndarray) and isinstance(self.y, np.ndarray):
            self.clf = self.clf.fit(self.x, self.y)
        else:
            raise ValueError("No input array or label array specified!")

    def predict(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self.clf:
            return self.clf.predict(x)
        else:
            raise Warning("No classifier has been created")

    def save_clf(self, path: str) -> None:
        if not path:
            raise ValueError("No save path specified.")
        if not path.endswith(".joblib"):
            raise ValueError("Path does not end with the .joblib extension")
        dump(self.clf, path)

    def load_clf(self, path: str) -> None:
        if not path:
            raise ValueError("No save path specified.")
        if not path.endswith(".joblib"):
            raise ValueError("Path does not end with the .joblib extension")
        self.clf = load(path)

    def set_clf(self, clf: object) -> None:
        if isinstance(clf, LinearRegression) or isinstance(clf, Ridge) or isinstance(clf, SVR):
            self.clf = clf
        else:
            raise Warning("Classifier is not a support sklearn type, no classifier was set")
