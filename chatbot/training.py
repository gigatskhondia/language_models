import random
import pickle
import os

import nltk
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from chatbot.conf import MODEL_WEIGHTS
from chatbot.intents import INTENT
from chatbot.dialog import AbstractML


class Training(AbstractML):
    ignore_words = {'?'}

    def pre_init(self):

        for intent, content in INTENT.items():
            for pattern in content['patterns']:
                # tokenize each word in the sentence
                w = nltk.word_tokenize(pattern)
                # add to our words list
                self.words.extend(w)
                # add to documents in our corpus
                self.documents.append((w, intent))
                # add to our classes list
                if intent not in self.classes:
                    self.classes.append(intent)

        self.words = sorted({self.stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words})
        self.classes = sorted(list(set(self.classes)))

        # create our training data
        self.training = []
        self.output = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in self.documents:

            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            self.training.append(
                [
                    self.bow(sentence=' '.join(doc[0]), words=self.words),
                    output_row,
                ]
            )

        # shuffle our features and turn into np.array
        random.shuffle(self.training)
        self.training = np.array(self.training)

        # create train and test lists
        self.train_x = list(self.training[:, 0])
        self.train_y = list(self.training[:, 1])

    def run(self):
        self.model.fit(self.train_x, self.train_y, n_epoch=10000, batch_size=12, show_metric=True)

    def save_model(self):

        new_dir_path = MODEL_WEIGHTS

        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)

        self.model.save(new_dir_path + 'model.tflearn')
        pickle.dump(
            {'words': self.words, 'classes': self.classes, 'train_x': self.train_x, 'train_y': self.train_y},
            open(new_dir_path + 'training_data', 'wb')
        )

    def cross_validation(self):
        cross_val = []
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(self.train_x, self.train_y, test_size=0.1, shuffle=True)
            self.model.fit(x_train, y_train, n_epoch=10000, batch_size=8, show_metric=True)
            y_prediction = self.model.predict(x_test)
            cross_val.append(f1_score(np.argmax(y_test, axis=1), np.argmax(y_prediction, axis=1), average='macro'))

        print('cross validation score is {}'.format(np.array(cross_val).mean()))
        return


if __name__ == "__main__":
    tr = Training()
    tr.run()
    tr.save_model()

