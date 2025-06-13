from task1.models.issue_type_classifier import IssueTypeClassifier
from task1.models.urgency_classifier import UrgencyClassifier
from task1.entity_extraction import extract_entities

# Load pre-trained models (for simplicity, assume they're trained and serialized, here we mock them)
issue_model = IssueTypeClassifier()
urgency_model = UrgencyClassifier()

# Fit dummy vectorizers to maintain code integrity (replace with load logic)
issue_model.load()
urgency_model.load()

def process_ticket(text):
    issue = issue_model.predict([text])[0]
    urgency = urgency_model.predict([text])[0]
    entities = extract_entities(text)
    return issue, urgency, entities
