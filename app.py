from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load AI models
chatbot = pipeline("text-generation", model="gpt2")
sentiment = pipeline("sentiment-analysis")

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    mood = ""
    if request.method == "POST":
        user_input = request.form["message"]
        response = chatbot(user_input, max_length=50, do_sample=True)[0]['generated_text']
        mood = sentiment(user_input)[0]['label']
    return render_template("index.html", response=response, mood=mood)

if __name__ == "__main__":
    app.run(debug=True)