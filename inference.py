import joblib
import re
import spacy
from nltk.corpus import stopwords

# Charger les modèles
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/logistic_regression_model.pkl')

# Stopwords
STOP_WORDS = set(stopwords.words('english'))
EXTRA_STOPWORDS = {'dear', 'hi', 'hello', 'please', 'thank', 'thanks'}
STOP_WORDS.update(EXTRA_STOPWORDS)

# Charger spaCy
nlp = spacy.load('en_core_web_sm')

def clean_text_basic(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return ' '.join(w for w in text.split() if w not in STOP_WORDS)

def lemmatize_spacy(text):
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc)

def full_preprocess(text):
    text = clean_text_basic(text)
    text = remove_stopwords(text)
    text = lemmatize_spacy(text)
    return text

def predict_ticket(text):
    text_clean = full_preprocess(text)
    text_tfidf = tfidf.transform([text_clean])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    return {
        'category': prediction,
        'confidence': max(probabilities),
        'cleaned_text': text_clean
    }

if __name__ == "__main__":
    # Test
    test = "My laptop won't turn on"
    result = predict_ticket(test)
    print(f"Ticket: {test}")
    print(f"Catégorie: {result['category']}")
    print(f"Confiance: {result['confidence']:.1%}")