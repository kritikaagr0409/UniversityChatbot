# import json
# import numpy as np
# from tensorflow import keras
# import colorama
# from flask import Flask, request, jsonify
# import pickle
#
# # Initialize Colorama
# colorama.init()
# from colorama import Fore, Style, Back
#
# # Load intents JSON
# with open("intents.json") as file:
#     data = json.load(file)
#
# # Load trained model
# model = keras.models.load_model('chat_model.keras')
#
# # Load tokenizer object
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# # Load label encoder object
# with open('label_encoder.pickle', 'rb') as enc:
#     lbl_encoder = pickle.load(enc)
#
# # Parameters
# max_len = 20
#
# # Initialize Flask app
# app = Flask(__name__)
#
#
# @app.route('/')
# def home():
#     return "Chatbot is up and running!"
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     message = request.json['message']
#     result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([message]),
#                                                                       truncating='post', maxlen=max_len))
#     tag = lbl_encoder.inverse_transform([np.argmax(result)])
#
#     for i in data['intents']:
#         if i['tag'] == tag:
#             response = np.random.choice(i['responses'])
#             break
#
#     return jsonify({"response": response})
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow import keras
import pickle
import json

app = Flask(__name__)

# Load the trained model and necessary objects
model = keras.models.load_model('chat_model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

with open('intents.json') as file:
    data = json.load(file)

# Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    message = request.json['message']
    max_len = 20
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([message]),
                                                                        truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            response = np.random.choice(i['responses'])
            break

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
