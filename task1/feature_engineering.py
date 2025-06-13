from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

    def fit_transform(self, X_train):
        return self.vectorizer.fit_transform(X_train)

    def transform(self, X):
        return self.vectorizer.transform(X)
