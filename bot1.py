from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import string
import random
import nltk

import os

nltk.download('punkt')
nltk.download('wordnet')

folder_path = "/home/sudip/Documents/nltk-chartbot/nltk-chartbot-main/bbcsport-fulltext/bbcsport/football"
row = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
        row.append(data)

# print(row)


row1 = " ".join(row).lower()
# print(row1)

sentance_tokens = nltk.sent_tokenize(row1)
word_tokenize = nltk.word_tokenize(row1)
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))


greet_inputs = ("hello", "Hello", "whatsup", "how are you?", "How are you?")
greed_responses = ("hi", "hay", "Hey There")


def greet(sentance):
    for word in sentance.split():
        if word.lower() in greet_inputs:
            return random.choice(greed_responses)


def response(user_responce):
    robo1_responce = ""
    Tfidfvc = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = Tfidfvc.fit_transform(sentance_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo1_responce = robo1_responce + "i am sorry.i can not understand"
        return robo1_responce
    else:
        robo1_responce = robo1_responce + sentance_tokens[idx]
        return robo1_responce

# setup voice to text
# speech to text
# start the section
import speech_recognition as sr

# r = sr.Recognizer()
# with sr.Microphone() as source:
#     print("Say something:")
#     audio = r.listen(source)

# try:
#     text = r.recognize_google(audio)
#     print("You said: " + text)
# except sr.UnknownValueError:
#     print("Speech Recognition could not understand audio")
# except sr.RequestError as e:
#     print("Could not request results from Speech Recognition service; {0}".format(e))

# end the section

import pyttsx3  # import the library

engine = pyttsx3.init()  # initialized the module
engine.setProperty("rete", 100)  # set speech speed
engine.setProperty("volume", 1.0)  # set volume level  between 0 and 1
engine.setProperty("voice", "english")  # set the voice language
# engine.say("Hello, how are you?")
# engine.runAndWait()

flag = True
print(
    "hello ! i am chartbot. start conversion using hello .for ending typing type bye !"
)
# add the voice module
engine.say(
    "hello ! i am chartbot. start conversion using hello .for ending typing type bye !"
)
engine.runAndWait()
while flag == True:
    # bind To The Html Page
    # engine.say(user_responce=input("Enter your input"))  # add voice module
    # engine.runAndWait()  # Run the voice module
    user_responce = input("Enter Your Input :-")  # normal termial input
    
    # r = sr.Recognizer()
    # with sr.Microphone() as source:
    #     print("Say something:")
    #     audio = r.listen(source)

    # try:
    #     text = r.recognize_google(audio)
    #     user_responce = text
    #     print("You said: " + text)
    # except sr.UnknownValueError:
    #     print("Speech Recognition could not understand audio")
    # except sr.RequestError as e:
    #     print("Could not request results from Speech Recognition service; {0}".format(e))

    
    user_responce = user_responce.lower()
    if user_responce != "bye":
        if user_responce == "thank you" or user_responce == "thanks":
            flag = False
            print("Welcome................")
            # add voice responce
            engine.say("Welcome................")

            engine.runAndWait()
        elif user_responce=="what is your name":
            print("i am a chart bot")
            engine.say("i am a chart bot based on nlp")
            engine.runAndWait()    
        else:
            if greet(user_responce) != None:
                print("Bot " + greet(user_responce))
                engine.setProperty("rete", 100)
                # add voice responce
                engine.say("Bot " + greet(user_responce))

                engine.runAndWait()
            else:
                sentance_tokens.append(user_responce)
                word_tokenize = word_tokenize + nltk.word_tokenize(user_responce)
                final = list(set(word_tokenize))
                print("Bot:", end="")
                print(response(user_responce))
                # add voice
                engine.say(response(user_responce))
                engine.runAndWait()
                sentance_tokens.remove(user_responce)
    else:
        flag = False
        print("Bot:goodbuy")
        # add voice responce
        engine.say("Bot:goodbuy")
        engine.runAndWait()
