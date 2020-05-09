import random
import pickle

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from chatbot.conf import MODEL_WEIGHTS
from chatbot.intents import INTENT


class AbstractML:
    stemmer = LancasterStemmer()

    def __init__(self):
        self.words = []
        self.classes = []
        self.documents = []
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.pre_init()

        # Build neural network
        model = Sequential()
        model.add(Dense(8, input_dim=self.train_x.shape[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.train_y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    @classmethod
    def clean_up_sentence(cls, sentence):
        return {cls.stemmer.stem(word.lower()) for word in nltk.word_tokenize(sentence)}

    @classmethod
    def bow(cls, sentence, words):
        return [
            w in cls.clean_up_sentence(sentence)
            for w in words
        ]

    def pre_init(self):
        raise NotImplementedError("Method is abstract")


class Classifier(AbstractML):

    error_threshold = 0.25
    context = {}

    def pre_init(self):
        data = pickle.load(open(MODEL_WEIGHTS + 'training_data', 'rb'))
        self.words = data['words']
        self.classes = data['classes']
        self.train_x = data['train_x']
        self.train_y = data['train_y']

    def __init__(self) -> None:
        super().__init__()
        # load json and create model
        json_file = open(MODEL_WEIGHTS + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(MODEL_WEIGHTS + "model.h5")
        self.model = loaded_model
        print("Loaded model from disk")

    def classify(self, sentence):
        bow = self.bow(sentence, self.words)
        if not any(bow):
            return None, 0.0

        # generate probabilities from the model
        # filter out predictions below a threshold
        results = [
            (self.classes[i], r)
            for i, r in enumerate(self.model.predict(np.array(bow).reshape(1, -1))[0])
            if r > self.error_threshold
        ]
        return max(results, key=lambda x: x[1]) if results else (None, 0.0)

    def response(self, sentence, user_id='123'):
        tag, _ = self.classify(sentence)
        if 'context_set' in INTENT[tag]:
            self.context[user_id] = INTENT[tag]['context_set']
        if 'context_filter' not in INTENT[tag] or \
            (user_id in self.context and 'context_filter' in INTENT[tag] and
             INTENT[tag]['context_filter'] == self.context[user_id]):
            return random.choice(INTENT[tag]['responses']) if tag else ''


if __name__ == "__main__":
    cl = Classifier()
    print(cl.response('we want to rent a moped'))
    print(cl.response('today'))

