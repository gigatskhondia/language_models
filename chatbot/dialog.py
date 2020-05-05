import random
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer

import tflearn
import tensorflow as tf

from chatbot.conf import MODEL_WEIGHTS
from chatbot.intents import INTENT


class AbstractML:
    stemmer = LancasterStemmer()

    def __init__(self):
        self.words = []
        self.classes = []
        self.documents = []
        self.train_x = []
        self.train_y = []
        self.pre_init()

        # Build neural network
        tf.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(self.train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.train_y[0]), activation='softmax')
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)

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
        self.model.load(MODEL_WEIGHTS + 'model.tflearn')

    def classify(self, sentence):
        bow = self.bow(sentence, self.words)
        if not any(bow):
            return None, 0.0

        # generate probabilities from the model
        # filter out predictions below a threshold
        results = [
            (self.classes[i], r)
            for i, r in enumerate(self.model.predict([bow])[0])
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
