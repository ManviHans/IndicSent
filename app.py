from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

sentiment_model = pipeline("sentiment-analysis")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_text = request.form["user_text"]
        if user_text.strip() != "":
            prediction = sentiment_model(user_text)[0]
            label = prediction["label"]
            score = prediction["score"]

            if label == "POSITIVE":
                emoji = "😊"
            elif label == "NEGATIVE":
                emoji = "😞"
            else:
                emoji = "😐"

            result = f"{emoji} {label} ({round(score*100, 2)}% confidence)"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    print("✅ Flask app is about to run...")
    app.run(debug=True)
