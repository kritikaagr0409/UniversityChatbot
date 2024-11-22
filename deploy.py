import json
import numpy as np
from tensorflow import keras
from flask import Flask, request, jsonify
import colorama
import pickle

colorama.init()
from colorama import Fore, Style

app = Flask(__name__)

# Load model and other necessary files
model = keras.models.load_model('chat_model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

with open("intents.json") as file:
    data = json.load(file)

max_len = 20

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the chatbot!"})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_message]),
                                                                       truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for intent in data['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            return jsonify({"response": response})

    return jsonify({"response": "I'm not sure how to respond to that."})

if __name__ == '__main__':
    print(Fore.YELLOW + "Starting Flask server..." + Style.RESET_ALL)
    app.run(debug=True)
