from sklearn.linear_model import LogisticRegression
from task1.feature_engineering import FeatureExtractor
import joblib
import os

class UrgencyClassifier:
    def __init__(self):
        self.model = LogisticRegression()
        self.fe = FeatureExtractor()

    def train(self, X_train, y_train):
        X_vec = self.fe.fit_transform(X_train)
        self.model.fit(X_vec, y_train)

    def predict(self, X):
        X_vec = self.fe.transform(X)
        return self.model.predict(X_vec)

    def save(self, path='urgency_model.pkl'):
        joblib.dump((self.model, self.fe), path)

    def load(self, path='urgency_model.pkl'):
        if os.path.exists(path):
            self.model, self.fe = joblib.load(path)
