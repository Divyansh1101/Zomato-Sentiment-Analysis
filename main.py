import os
import sys
import pandas as pd
import numpy as np

print("Starting Zomato Sentiment Analysis...")

DATA_PATH = os.path.join("data", "zomato_reviews.csv")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------- Data loading / fallback -------
if not os.path.exists(DATA_PATH):
    print("Dataset not found, using synthetic data ->", DATA_PATH)
    os.makedirs("data", exist_ok=True)
    synthetic = pd.DataFrame({
        "review": [
            "Loved the food and the ambiance!",
            "Terrible service and cold pizza.",
            "Great place for family dinners.",
            "The biryani was bland and overpriced.",
            "Amazing desserts, will visit again!",
            "Waited too long, very disappointing.",
            "Friendly staff and tasty burgers.",
            "Not worth the money."
        ],
        "sentiment": ["positive","negative","positive","negative","positive","negative","positive","negative"]
    })
    synthetic.to_csv(DATA_PATH, index=False)

df = pd.read_csv(DATA_PATH)

# ------- Basic preprocessing -------
def clean_text(s):
    if not isinstance(s, str): 
        return ""
    s = s.lower()
    return s

df["clean"] = df["review"].apply(clean_text)

# Try to use NLP tools, but be resilient if missing
tokenizer_ok = True
stopwords_ok = True
try:
    import nltk  # type: ignore
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords  # type: ignore
    from nltk.tokenize import word_tokenize  # type: ignore
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    print("NLTK not fully available, falling back to naive tokenization:", e)
    tokenizer_ok = False
    stopwords_ok = False
    STOPWORDS = set(["the","a","an","and","is","are","to","for","of","in","on","with","very","too"])

def tokenize(text):
    if tokenizer_ok:
        try:
            return [w for w in word_tokenize(text) if w.isalpha()]
        except Exception as e:
            print("Tokenizer error, falling back to split:", e)
    return [w for w in text.split() if w.isalpha()]

def remove_stopwords(tokens):
    if stopwords_ok:
        return [w for w in tokens if w not in STOPWORDS]
    return [w for w in tokens if w not in STOPWORDS]

df["tokens"] = df["clean"].apply(tokenize)
df["tokens_nostop"] = df["tokens"].apply(remove_stopwords)

# ------- WordCloud -------
wordcloud_path = os.path.join(RESULTS_DIR, "wordcloud.png")
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    text_blob = " ".join([" ".join(toks) for toks in df["tokens_nostop"]])
    if not text_blob.strip():
        text_blob = "food tasty service good bad delicious ambiance staff price wait"
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(wordcloud_path, dpi=150)
    plt.close()
    print(f"Saved word cloud -> {wordcloud_path}")
except Exception as e:
    print("WordCloud generation failed, skipping image:", e)

# ------- ML: Logistic Regression -------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

X_text = df["clean"].fillna("")
y = df["sentiment"].astype(str).str.lower()

vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
report = classification_report(y_test, pred)

eval_path = os.path.join(RESULTS_DIR, "evaluation.txt")
with open(eval_path, "w", encoding="utf-8") as f:
    f.write("Zomato Sentiment Analysis\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)
print(f"Saved evaluation -> {eval_path}")

print("Done.")
