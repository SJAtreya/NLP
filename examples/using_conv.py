from __future__ import print_function
from preprocessor import Dataset
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import optimizers
import numpy as np
import keras, gensim, logging
from keras.callbacks import RemoteMonitor
from gensim_training import WordTrainer
import os
from os import listdir
from os.path import isfile, join
from keras.utils import plot_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
remote = RemoteMonitor()

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 750
EMBEDDING_DIM = 10
VALIDATION_SPLIT = 0.2
max_features = 750
maxlen = 150 # cut texts after this number of words (among top max_features most common words)
batch_size = 512
num_classes = 4
print('Loading data...')
ds = Dataset()
X_all = ds.X_all_sent
Y_all = ds.Y_all
print('loaded word2vec...')
wordTrainer = WordTrainer()
print('training sentences...')
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_all)
sequences = tokenizer.texts_to_sequences(X_all)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
padded_sequences = sequence.pad_sequences(sequences, maxlen=maxlen)
labels = keras.utils.to_categorical(np.asarray(Y_all))
print ("Labels created...")
x_train, x_test, y_train, y_test = train_test_split(
                                   padded_sequences,labels,test_size=0.2, random_state=42)
print('train test split complete')
test_data = open('test/test.csv').read().split('\n')
print(test_data)
test_tokenizer = Tokenizer(num_words=max_features)
test_tokenizer.fit_on_texts(test_data)
test_sequences = sequence.pad_sequences(test_tokenizer.texts_to_sequences(test_data),maxlen=maxlen)
print ("Additional test data loaded...")

embeddings_index = {}
f = open(os.path.join("", 'trained_vectors.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:len(values)], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
#print (embedding_matrix)
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    #print ("embedding vector", embedding_vector)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('Build model...')


embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_shape=(maxlen,),
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
#x = Conv1D(128, 5, activation='relu')(x)
#x = MaxPooling1D(5)(x)
#x = Conv1D(128, 5, activation='relu')(x)
#x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(4, activation='softmax')(x)
optimizer = optimizers.Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=100,
          validation_data=(x_test, y_test), callbacks=[remote])
scores = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(test_sequences, batch_size=128, verbose=1)
for prediction in predictions:
    predicted_class = np.argmax(prediction)
    print (predicted_class)
print("For x_test")
predictions = model.predict(x_test, batch_size=128, verbose=1)
for prediction in predictions:
    predicted_class = np.argmax(prediction)
    print (predicted_class)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#plot_model(model, to_file='model.png')