import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class IssueTypeClassifier:
    def __init__(self, model_path='task1/models/issue_type_model.pkl'):
        self.model_path = model_path
        self.model = None

    def build_pipeline(self):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
            ('clf', LogisticRegression(solver='liblinear', random_state=42))
        ])
        return pipeline

    def train(self, X, y):
        print("[IssueTypeClassifier] Training started...")
        self.model = self.build_pipeline()
        self.model.fit(X, y)
        print("[IssueTypeClassifier] Training complete.")
        self.save_model()
    
    def evaluate(self, X, y):
        if not self.model:
            raise ValueError("Model not trained yet.")
        preds = self.model.predict(X)
        return classification_report(y, preds)

    def save_model(self):
        if self.model:
            joblib.dump(self.model, self.model_path)
            print(f"[IssueTypeClassifier] Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"[IssueTypeClassifier] Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"No model found at {self.model_path}")

    def predict(self, ticket_text: str) -> str:
        if not self.model:
            self.load_model()
        return self.model.predict([ticket_text])[0]
