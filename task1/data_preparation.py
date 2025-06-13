import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_excel(path)
    return df

def split_data(df):
    X = df['ticket_text']
    y_issue = df['issue_type']
    y_urgency = df['urgency_level']
    X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test = train_test_split(
        X, y_issue, y_urgency, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test
