from __future__ import print_function
from preprocessor import Dataset
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
import numpy as np
import keras, gensim, logging
from keras import callbacks
from gensim_training import WordTrainer
import os
from os import listdir
from os.path import isfile, join

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
remote = callbacks.RemoteMonitor(root='http://localhost:9000')

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 750
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
max_features = 750
maxlen = 750 # cut texts after this number of words (among top max_features most common words)
batch_size = 512
num_classes = 4
print('Loading data...')
ds = Dataset()
X_all = ds.X_all_sent
Y_all = ds.Y_all
print('loaded word2vec...')
wordTrainer = WordTrainer()
print('training sentences...')
x_train, x_test, y_train, y_test = train_test_split(
                                   X_all,Y_all,test_size=0.3, random_state=42)
print('train test split complete')
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_sequences(x_train)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

embeddings_index = {}
f = open(os.path.join("", 'trained_vectors.txt'))
for line in f:
    values = line.split()
    word = ' '.join(values[0:len(values)-100])
    coefs = np.asarray(values[len(values)-100:len(values)], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(X_all))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for i, sent in enumerate(X_all):
    embedding_vector = embeddings_index.get(sent)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print (embedding_matrix)
print('Build model...')
model = Sequential()
model.add(Embedding(num_words, 100,weights=[embedding_matrix],input_shape=(maxlen,), trainable=False))
#model.add(LSTM(32, dropout=0.001, recurrent_dropout=0.001, return_sequences=True))
#model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(4, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test), callbacks=[remote])
scores = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
#test_data = open('test/test.csv').read().split('\n')
#print(test_data)
#tests = tokenizer.texts_to_sequences(test_data)
predictions = model.predict(x_test, batch_size=128)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
rounded = [round(x[0]) for x in predictions]
print(x_test)
print(predictions)