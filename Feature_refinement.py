# 1  Importiere benötigte Bibliotheken
import re                                      # 1
import numpy as np                             # 2
import pandas as pd                            # 3
from sklearn.base import BaseEstimator, TransformerMixin  # 4
from sklearn.feature_extraction.text import TfidfVectorizer  # 5
from sklearn.pipeline import FeatureUnion, Pipeline         # 6
from sklearn.preprocessing import StandardScaler            # 7
from sklearn.model_selection import train_test_split        # 8
from sklearn.linear_model import LogisticRegression         # 9
from sklearn.metrics import classification_report           # 10
from sklearn.utils.validation import check_is_fitted        # 11

# 12  Beispiel-Daten (du ersetzt das später durch dein echtes DataFrame)
data = [
    ("Hey, kannst du mir bis morgen das Dokument schicken? Danke!", 0),  # kein Spam
    ("Gewinne jetzt 1.000.000€!!! Klicke hier: http://spam.example", 1), # Spam
    ("Meeting morgen 10 Uhr — Agenda im Anhang.", 0),
    ("Sonderangebot nur heute!!! billig viagra bestellen www.pharma.example", 1),
    ("Kurze Frage: bist du im Büro?", 0),
    ("LAST WARNING: your account will be closed, click http://phish.example", 1)
]  # 13

df = pd.DataFrame(data, columns=["text", "label"])
#Erstellt Pandas DataFrame, mit den Spalten "Text" und "Label"

# 15  Eigener Transformer: extrahiert einfache handgemachte (numeric) Features aus dem Text
class TextStatsTransformer(BaseEstimator, TransformerMixin):  # 16
    def __init__(self):                                      # 17
        pass                                                # 18

    def fit(self, X, y=None):                                # 19
        return self                                         # 20

    def transform(self, X):                                 # 21
        # X ist eine Liste/Series von Texten. Wir geben ein numpy array mit numerischen Features zurück.
        Features = []                                       # 22
        for doc in X:                                       # 23
            # Grundlegende Zähl-Features:
            length = len(doc)                               # 24  # Zeichenanzahl
            exclam = doc.count('!')                         # 25  # Anzahl Ausrufezeichen
            urls = len(re.findall(r"http[s]?://|www\.", doc))  # 26  # einfache URL-Erkennung
            digits = sum(c.isdigit() for c in doc)          # 27  # Ziffern zählen
            upper_words = sum(1 for w in doc.split() if w.isupper())  # 28  # GROSS geschriebene Wörter
            # Verhältnis Großbuchstaben (0..1)
            upper_ratio = (sum(1 for c in doc if c.isupper()) / max(1, length))  # 29
            features.append([length, exclam, urls, digits, upper_words, upper_ratio])  # 30
        return np.array(features)                           # 31

# 32  Zwei TF-IDF Vektorisierer: Wort-ngrams und char-ngrams (fangen unterschiedliche Muster ein)
tfidf_word = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)   # 33
tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1, max_df=0.95)  # 34

# 35  FeatureUnion kombiniert die unterschiedlichen Feature-Quellen (Wörter, Character, Handgemachte)
features = FeatureUnion([
    ('word_tfidf', tfidf_word),   # 36
    ('char_tfidf', tfidf_char),   # 37
    ('text_stats', TextStatsTransformer())  # 38
])  # 39

# 40  Pipeline: erst alle Features bauen, dann skalieren (kein mean-centering wegen Sparsität), dann Klassifizierer
pipe = Pipeline([
    ('features', features),                                    # 41
    ('scaler', StandardScaler(with_mean=False)),               # 42
    ('clf', LogisticRegression(solver='liblinear', random_state=42))  # 43
])  # 44

# 45  Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.33, random_state=42)  # 46

# 47  Modell trainieren
pipe.fit(X_train, y_train)  # 48

# 49  Vorhersage und Evaluation
y_pred = pipe.predict(X_test)  # 50
print(classification_report(y_test, y_pred))  # 51

# 52  OPTIONAL: so kannst du sehen, welche numerischen Features wir gebaut haben (für einen Blick)
# Wir rufen transform() der 'text_stats' Komponente direkt auf:
text_stats = features.transformer_list[2][1]  # 53  # greift auf TextStatsTransformer zu
stats_matrix = text_stats.transform(X_test)   # 54
stats_df = pd.DataFrame(stats_matrix, columns=["length","exclam","urls","digits","upper_words","upper_ratio"])  # 55
print("\nHandgemachte Features (Testset):")  # 56
print(stats_df)  # 57
