from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import string
import random
import nltk
import os

# import pyttsx3
# import speech_recognition as sr

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

folder_path = "./bbcsport-fulltext/bbcsport/cricket"
path="pwd"
print(path)
row = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
        row.append(data)

row1 = " ".join(row).lower()

sentance_tokens = nltk.sent_tokenize(row1)
word_tokenize = nltk.word_tokenize(row1)
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))


greet_inputs = ("hello", "Hello", "whatsup", "how are you?", "How are you?","hi","Hi","HI","Whats Up","Whats up")
greed_responses = ("hi", "hay", "Hey There")


def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greed_responses)


def response(user_response):
    robo1_response = ""
    Tfidfvc = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = Tfidfvc.fit_transform(sentance_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo1_response = robo1_response + "I am sorry, I cannot understand."
        return robo1_response
    else:
        robo1_response = robo1_response + sentance_tokens[idx]
        return robo1_response


# engine = pyttsx3.init()
# engine.setProperty("rate", 150)
# engine.setProperty("volume", 1.0)
# engine.setProperty("voice", "english")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_response = request.form['user_input']
    bot_response = process_user_input(user_response)
    return bot_response


def process_user_input(user_input):
    flag = True
    while flag == True:
        user_input = user_input.lower()

        if user_input != "bye":
            if user_input == "thank you" or user_input == "thanks":
                response1 = "Welcome..."
            elif user_input == "what is your name" or user_input=="who are you":
                response1 = "I am a chatbot."
            else:
                if greet(user_input) is not None:
                    response1 =  greet(user_input)
                    print("Bot: " + greet(user_input))
                else:
                    sentance_tokens.append(user_input)
                    word_tokenize1 = word_tokenize + nltk.word_tokenize(user_input)
                    final = list(set(word_tokenize1))
                    response1 =  response(user_input)  # Fix the variable name here
                    print(response1)
                    sentance_tokens.remove(user_input)
        else:
            flag = False
            response1 = " Goodbye"

        return response1


if __name__ == '__main__':
    # using docker deployment add porr and host
    app.run(debug=True,host='0.0.0.0', port=5000)
