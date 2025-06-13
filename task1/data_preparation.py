import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_data(file_path):
    df = pd.read_excel(file_path)
    df.dropna(subset=['ticket_text', 'issue_type', 'urgency_level', 'product'], inplace=True)
    return df

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def preprocess_text(text):
    tokens = normalize_text(text).split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def prepare_dataset(df):
    df['clean_text'] = df['ticket_text'].apply(preprocess_text)
    return df
