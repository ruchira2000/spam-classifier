import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import clean_text
import pickle

df = pd.read_csv("../data/sms.tsv", sep="\t", header=None, names=["label", "message"])
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
df["clean_message"] = df["message"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)

print(f"Total messages: {len(df)}")
print(f"Training set: {len(X_train)}")
print(f"Testing set: {len(X_test)}")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Training matrix shape: {X_train_vec.shape}")

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_vec, y_train)
lr_preds = log_reg.predict(X_test_vec)

svm = LinearSVC(class_weight="balanced")
svm.fit(X_train_vec, y_train)
svm_preds = svm.predict(X_test_vec)

for name, preds in [("Logistic Regression", lr_preds), ("Linear SVM", svm_preds)]:
    print(f"\n{name}")
    print(f"  Accuracy:  {accuracy_score(y_test, preds)*100:.2f}%")
    print(f"  Precision: {precision_score(y_test, preds)*100:.2f}%")
    print(f"  Recall:    {recall_score(y_test, preds)*100:.2f}%")
    print(f"  F1 score:  {f1_score(y_test, preds)*100:.2f}%")

with open("../models/model.pkl", "wb") as f:
    pickle.dump(svm, f)

with open("../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved to /models")