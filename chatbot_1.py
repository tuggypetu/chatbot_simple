"""Python chatbot"""
import nltk
import numpy
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
import pandas as pd
import tflearn
# # TODOne: Install with MAC instructions only!!!
import tensorflow as tf

# tf.compat.v1.disable_resource_variables()
# print(tf.__version__)

import random
import json
import pickle

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)
# print(data["intents"])

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)
    # print(words)
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    # print(out_empty)
    # print(docs_x)
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        # print(wrds)
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        # print(bag)

        output_row = out_empty[:]
        # print(output_row)
        output_row[labels.index(docs_y[x])] = 1
        # print(output_row)
        # print(docs_y[x])
        # print(output_row[labels.index(docs_y[x])])

        training.append(bag)
        output.append(output_row)
    # print(docs_x)
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# print(training)
# print(output)
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words_):
    """"""
    bag = [0 for _ in range(len(words_))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    """"""
    print("Start talking with the bot! (type quit to stop)")
    inp = input("You: ")
    while inp.lower() != "quit":
        results = model.predict([bag_of_words(inp, words)])[0]
        print(results)
        results_index = np.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.7:
            response = 'Nothing'
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    response = random.choice(responses)
            print("Bot:", response)
        else:
            print("Sorry, did not understand, please try again!")
        inp = input("You: ")


chat()
