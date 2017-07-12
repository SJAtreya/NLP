import gensim, logging
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import word2vec
from gensim.models import Phrases
from sklearn.manifold import TSNE

data_path = 'gensim_training'
labels = [f for f in listdir(data_path) if isfile(join(data_path, f))]
class WordTrainer(object):
    def __init__(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        Y_all = []
        x_sents=[]
        for label in labels:
            with open('gensim_training/'+label,'r') as f:
                for line in f:
                    x_sents.append(line.split())
            Y_all.append(label.split(".")[0])
        #sentences = word2vec.LineSentence('for_lstm/1.txt')
        bigram_transformer = gensim.models.Phrases(x_sents)
        self.model = word2vec.Word2Vec(bigram_transformer[x_sents], size=10)
        self.Y_all = Y_all
        #self.model.build_vocab(x_sents)  # can be a non-repeatable, 1-pass generator
        #self.model.train(x_sents)  # can be a non-repeatable, 1-pass generator
        self.word_vectors = self.model.wv
        self.word_vectors.evaluate_word_pairs(os.path.join('.', 'gensim_training','wordsimilarity.txt'))
        self.word_vectors.accuracy(os.path.join('.', 'gensim_training', 'questions-words.txt'))
        self.word_vectors.save_word2vec_format("trained_vectors.txt")
        print("Similarity between schedule and bone:", self.model.similarity('schedule','bone'))
        print("Similarity between perform and bone:", self.model.similarity('perform','bone'))
        X = self.model[self.word_vectors.vocab]
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(X)
        plt.scatter(Y[:, 0], Y[:, 1])
        for label, x, y in zip(self.word_vectors.vocab, Y[:, 0], Y[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.show()		
        del self.model