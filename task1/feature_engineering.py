from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

def extract_features(df):
    tfidf = TfidfVectorizer(max_features=500)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])
    df['ticket_length'] = df['ticket_text'].apply(len)
    df['sentiment'] = df['ticket_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return tfidf_matrix, df[['ticket_length', 'sentiment']]
