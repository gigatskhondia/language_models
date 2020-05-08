import nltk
import random
from nltk.stem.lancaster import LancasterStemmer

from chatbot_without_ai.intents import INTENT


class Classifier:
    context = {}
    ignore_words = {'?', 's'}
    stemmer = LancasterStemmer()

    def __init__(self):
        self.corpus_words = {}
        self.class_words = {}
        self.classes = []

    def pre_processing(self):
        for intent, content in INTENT.items():
            self.classes.append(intent)

        for intent in self.classes:
            self.class_words[intent] = []

        for intent, content in INTENT.items():
            for pattern in content['patterns']:
                # tokenize each word in the sentence
                words = nltk.word_tokenize(pattern)
                # add to our words list
                for word in words:
                    if word not in self.ignore_words:
                        stemmed_word = self.stemmer.stem(word.lower())

                        if stemmed_word not in self.corpus_words:
                            self.corpus_words[stemmed_word] = 1
                        else:
                            self.corpus_words[stemmed_word] += 1

                        self.class_words[intent].extend([stemmed_word])

    def calculate_class_score(self, sentence, class_name):
        score = 0
        # tokenize each word in our new sentence
        for word in nltk.word_tokenize(sentence):
            # check to see if the stem of the word is in any of our classes
            if self.stemmer.stem(word.lower()) in self.class_words[class_name]:
                # treat each word with relative weight
                score += (1 / self.corpus_words[self.stemmer.stem(word.lower())])

        return score

    def classify(self, sentence):
        high_class = None
        high_score = 0
        # loop through our classes
        for cl in self.class_words.keys():
            # calculate score of sentence for each class
            score = self.calculate_class_score(sentence, cl)
            # keep track of highest score
            if score > high_score:
                high_class = cl
                high_score = score

        return high_class, high_score

    def response(self, sentence):
        tag, _ = self.classify(sentence)
        return random.choice(INTENT[tag]['responses']) if tag else ''


if __name__ == "__main__":
    cls = Classifier()
    cls.pre_processing()
    print(cls.response('hi'))
    print(cls.response('we want to rent a moped'))
    print(cls.response('today'))

