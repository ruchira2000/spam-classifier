# Email & SMS Spam Classifier

From a raw text message to a spam or ham decision, in real time, through a trained ML pipeline served by a Flask web app.

This project takes 5,572 real SMS messages, cleans and vectorizes the text, trains and compares two classification models, and serves the best one through a live web interface where you can paste any message and get an instant prediction.

## How it works

```
Raw message
   |
   |-- 1. Text cleaning (lowercase, strip links/emails/numbers, remove stopwords, stem with PorterStemmer)
   |-- 2. TF-IDF vectorization (unigrams + bigrams, 5000 features)
   |-- 3. Model comparison: Logistic Regression vs Linear SVM
   |-- 4. Best model (Linear SVM) saved and loaded into a Flask app
   |
   |-- Flask web app: paste a message, get a live prediction
```

## Tech stack

- **Python**, pandas, scikit-learn
- **NLTK** for stopword removal and stemming
- **TF-IDF** for turning text into features
- **Logistic Regression** and **Linear SVM** (compared, best one kept)
- **Flask** for the web app, **HTML/CSS/JS** for the interface

## Dataset

[SMS Spam Collection](https://doi.org/10.24432/C5CC84), a public dataset of 5,572 real SMS messages from the UCI Machine Learning Repository, labeled `ham` or `spam` (4,825 ham, 747 spam). Included in this repo at `data/sms.tsv`, no download needed.

## Results

Both models were trained on an 80/20 stratified split (4,457 training, 1,115 testing) and evaluated on the held out test set.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 97.67% | 88.68% | 94.63% | 91.56% |
| **Linear SVM (used)** | **98.92%** | **96.60%** | **95.30%** | **95.95%** |

Linear SVM won across every metric. This lines up with how the two algorithms work: SVM looks for the widest possible margin between spam and ham in the data, which tends to make it more conservative and confident, fewer false alarms, and here it also caught more real spam.

Class imbalance (87% ham, 13% spam) was handled with `class_weight="balanced"` rather than oversampling, so the model could not take the shortcut of just guessing ham most of the time and still scoring well.

## How to run it

1. Clone the repo and set up the environment:
```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
```
2. Download the NLTK stopwords data (one time):
```
   python -c "import nltk; nltk.download('stopwords')"
```
3. Train the model (already trained and saved, run this to reproduce it):
```
   cd src
   python train.py
```
4. Start the web app:
```
   python app.py
```
   Then open http://127.0.0.1:5000

## Project structure

```
spam-classifier/
├── data/sms.tsv          raw dataset
├── src/
│   ├── preprocess.py     text cleaning function
│   └── train.py          training, evaluation, saving the model
├── models/                saved model and vectorizer (pickled)
├── static/style.css       app styling
├── templates/index.html   app frontend
├── app.py                 Flask app
└── requirements.txt
```

## Honest notes on results

This dataset skews toward SMS style spam (prize scams, premium rate numbers) rather than modern phishing wording, so a message like "your account has been suspended, verify here" sometimes gets missed since that phrasing is closer to email phishing than the SMS spam this model was trained on.

Building the text cleaner also surfaced two real, worth mentioning issues: stripping punctuation with a blunt delete initially glued words together across removed brackets and periods (fixed by replacing punctuation with a space instead of deleting it), and removing stopwords too aggressively occasionally stripped the "t" out of a split contraction like "don't", turning it into a stopword itself and quietly dropping a negation. Both are small but real examples of how easy it is for text preprocessing to introduce subtle bugs that don't throw errors, they just quietly change your data.

## What this project demonstrates

- End to end ML pipeline: raw text to a deployed, working prediction interface
- Comparing multiple models on the same data with proper train/test evaluation, not just picking one and hoping
- Handling real world data issues: class imbalance, noisy text, preprocessing edge cases
- Deploying a model behind a usable web interface, not leaving it in a notebook

## Author

Ruchira Surendra, Data Analyst, Melbourne
[LinkedIn](https://www.linkedin.com/in/ruchira-surendra)