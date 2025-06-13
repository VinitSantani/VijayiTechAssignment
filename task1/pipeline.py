from task1.models.issue_type_classifier import IssueTypeClassifier
from task1.models.urgency_classifier import UrgencyClassifier
from task1.data_preparation import preprocess_text
from task1.feature_engineering import extract_features
from task1.entity_extraction import extract_entities

class TicketPipeline:
    def __init__(self, issue_model, urgency_model, vectorizer, product_list, complaint_keywords):
        self.issue_model = issue_model
        self.urgency_model = urgency_model
        self.vectorizer = vectorizer
        self.product_list = product_list
        self.complaint_keywords = complaint_keywords

    def predict(self, text):
        clean = preprocess_text(text)
        vectorized = self.vectorizer.transform([clean])
        issue_type = self.issue_model.predict(vectorized)[0]
        urgency = self.urgency_model.predict(vectorized)[0]
        entities = extract_entities(text, self.product_list, self.complaint_keywords)
        return {
            'issue_type': issue_type,
            'urgency_level': urgency,
            'entities': entities
        }
