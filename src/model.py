from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'classifier.joblib'
DATA = ROOT / 'data' / 'transactions.csv'

def train(path=DATA, output=MODEL):
    df = pd.read_csv(path)
    if df.empty or df['category'].nunique() < 2:
        raise ValueError('Training data must contain at least two categories')
    clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
                    ('classifier', LogisticRegression(max_iter=1200, random_state=42))])
    clf.fit(df['description'].fillna(''), df['category'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output)
    return clf

def predict(descriptions, model_path=MODEL):
    if isinstance(descriptions, str):
        descriptions = [descriptions]
    if not descriptions or any(not isinstance(x,str) or not x.strip() for x in descriptions):
        raise ValueError('Provide nonempty transaction descriptions')
    clf = joblib.load(model_path)  # Only load trusted model artifacts.
    probabilities = clf.predict_proba(descriptions)
    labels = clf.classes_
    return [{'description': text, 'category': str(labels[row.argmax()]),
             'confidence': round(float(row.max()), 4),
             'needs_review': bool(row.max() < 0.65)}
            for text, row in zip(descriptions, probabilities)]
