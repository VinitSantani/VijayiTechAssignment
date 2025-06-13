import os
from task1.data_preparation import load_data, prepare_dataset
from task1.feature_engineering import extract_features
from task1.models.issue_type_classifier import IssueTypeClassifier
from task1.models.urgency_classifier import UrgencyClassifier
from task1.pipeline import TicketPipeline
from sklearn.feature_extraction.text import TfidfVectorizer

PRODUCT_LIST = ['widget', 'gadget', 'device', 'unit']
COMPLAINT_KEYWORDS = ['broken', 'late', 'error', 'damaged', 'defective']

def main():
    df = load_data("task1/data/ai_dev_assignment_tickets_complex_1000.xlsx")
    df = prepare_dataset(df)
    tfidf_vectorizer = TfidfVectorizer(max_features=500)
    X_text = tfidf_vectorizer.fit_transform(df['clean_text'])

    issue_classifier = IssueTypeClassifier()
    urgency_classifier = UrgencyClassifier()

    issue_classifier.train(X_text, df['issue_type'])
    urgency_classifier.train(X_text, df['urgency_level'])

    pipeline = TicketPipeline(issue_classifier, urgency_classifier, tfidf_vectorizer, PRODUCT_LIST, COMPLAINT_KEYWORDS)

    test_text = "My device was broken and delivered late on 12/05/2024."
    result = pipeline.predict(test_text)
    print(result)

if __name__ == '__main__':
    main()
