import joblib
import os

class IssueTypeClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer()
        X_vec = self.vectorizer.fit_transform(X)

        self.model = RandomForestClassifier()
        self.model.fit(X_vec, y)

    def predict(self, text):
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model or vectorizer not loaded.")
        X_vec = self.vectorizer.transform([text])
        return self.model.predict(X_vec)[0]

    def save(self, path="models/issue_type_model.joblib"):
        joblib.dump((self.model, self.vectorizer), path)

    def load(self, path="models/issue_type_model.joblib"):
        if os.path.exists(path):
            self.model, self.vectorizer = joblib.load(path)
        else:
            raise FileNotFoundError(f"No saved model found at {path}")
