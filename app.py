import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request
import json
import random

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'

# Load intent recognition model and tokenizer
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Functions for chatbot
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    try:
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    except Exception as e:
        print(f"Prediction Error: {e}")
        return []

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(msg):
    try:
        if not msg.strip():
            return "Please provide some input."

        # Directly process the input as English
        return getResponse(predict_class(msg, model), intents)
    except Exception as e:
        print(f"Chatbot Response Error: {e}")
        return "Error generating response."

# Flask routes
@app.route("/")
def home():
    welcome_message = "Welcome to Talk2Heal, your companion, a safe and supportive space where you can share your thoughts and feelings without fear of judgement."
    return render_template("index.html", welcome_message=welcome_message)


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(f"User message: {userText}")
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run(debug=True)
