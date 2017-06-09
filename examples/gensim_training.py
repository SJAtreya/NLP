import gensim, logging
import os
from os import listdir
from os.path import isfile, join


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_path = 'for_lstm'
labels = [f for f in listdir(data_path) if isfile(join(data_path, f))]
class WordTrainer(object):
    def __init__(self):
        Y_all = []
        x_sents=[]
        for label in labels:
            x_file = open('for_lstm/'+label)
            x_sents.append(x_file.read().split('\n'))
            Y_all.append(label.split(".")[0])
        self.model = gensim.models.Word2Vec(x_sents, iter=5)
        self.Y_all = Y_all
        #model.build_vocab(sentences)  # can be a non-repeatable, 1-pass generator
        self.model.train(x_sents)  # can be a non-repeatable, 1-pass generator
        self.word_vectors = self.model.wv
        self.word_vectors.save_word2vec_format("trained_vectors.txt")
        del self.model