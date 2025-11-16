from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Load model + vectorizer
model = pickle.load(open("../sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("../tfidf.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        review = request.form["review"]
        clean = clean_text(review)
        vector = tfidf.transform([clean])
        prediction = model.predict(vector)[0]

        if prediction == "positive":
            result = "Positive 😄"
        else:
            result = "Negative 😡"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
