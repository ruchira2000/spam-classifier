import pickle
import sys
sys.path.insert(0, "src")
from preprocess import clean_text

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    label = "spam" if prediction == 1 else "ham"
    return jsonify({"message": message, "label": label})


if __name__ == "__main__":
    app.run(debug=True)