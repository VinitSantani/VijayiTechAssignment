from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class IssueTypeClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        print("Issue Type Classifier Report:\n", classification_report(y_test, preds))

    def predict(self, X):
        return self.model.predict(X)
