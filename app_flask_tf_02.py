# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
import ssl, os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# chat initialization

# Load and preprocess training data
model = load_model("chatbot_model.h5")
data_file = open("intents.json").read()
intents = json.loads(data_file)
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Tokenizer setup
tokenizer = Tokenizer(num_words=len(words))
tokenizer.fit_on_texts(words)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]

    # Load and process the intents JSON file
    data_file = open("intents.json").read()
    intents = json.loads(data_file)
    
    # Check if the message contains any tag
    for intent in intents["intents"]:
        tag = intent["tag"]
        if tag.lower() in msg.lower():
            # If a tag is found, return all responses for the corresponding tag
            responses = intent["responses"]
            
            # Check if any response contains a link
            for i, response in enumerate(responses):
                if "<a href=" in response:
                    # Add target="_blank" to open the link in a new tab
                    responses[i] = response.replace("<a href=", "<a target='_blank' href=")
            
            return " ".join(responses)

    # If no match is found, provide a default response
    return "Sorry, I couldn't understand that."

    # Rest of your existing code
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = tokenizer.texts_to_sequences([sentence])
    sentence_words = pad_sequences(sentence_words, maxlen=len(words))
    return sentence_words[0]

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words))
    for i in sentence_words:
        if i != 0:
            bag[i-1] = 1
            if show_details:
                print(f"found in bag: {words[i-1]}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

