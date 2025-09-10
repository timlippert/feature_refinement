from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 1. Mini-Korpus: nur drei "Dokumente" (z. B. E-Mails)
docs = [
    "buy cheap viagra now",      # Dokument 0
    "buy cheap offer now",       # Dokument 1
    "meeting tomorrow morning"   # Dokument 2
]

# 2. TF-IDF Vektorisierer definieren
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),   # Unigramme und Bigramme
    min_df=1,            # Feature muss in mind. 1 Dokument vorkommen
    max_df=0.95          # Features, die in mehr als 95% der Dokumente vorkommen, werden ignoriert
)

# 3. Fit auf dem gesamten Korpus (alle drei Dokumente zusammen!)
X = vectorizer.fit_transform(docs)

# 4. Vokabular anzeigen
print("Vokabular (alle erkannten N-Gramme):")
print(vectorizer.get_feature_names_out())

# 5. TF-IDF-Matrix als DataFrame anzeigen
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix (Zeilen = Dokumente, Spalten = N-Gramme):")
print(df)

# 6. Beispiel: nur ein Dokument transformieren
new_doc = ["cheap cheap cheap viagra"]
vec_new = vectorizer.transform(new_doc)
print("\nVektor für neues Dokument:")
print(pd.DataFrame(vec_new.toarray(), columns=vectorizer.get_feature_names_out()))
