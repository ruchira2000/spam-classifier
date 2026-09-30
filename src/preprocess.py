import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " httpaddr ", text)
    text = re.sub(r"\S+@\S+", " emailaddr ", text)
    text = re.sub(r"\d+", " numbr ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)