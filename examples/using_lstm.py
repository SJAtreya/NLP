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
import keras

max_features = 20000
maxlen = 200  # cut texts after this number of words (among top max_features most common words)
batch_size = 512
num_classes = 4
print('Loading data...')
ds = Dataset()
X_all = ds.X_all_sent
Y_all = ds.Y_all
x_train, x_test, y_train, y_test = train_test_split(
                                   X_all,Y_all,test_size=0.1, random_state=42)
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

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128,input_shape=(maxlen,)))
model.add(LSTM(32, dropout=0.001, recurrent_dropout=0.001, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(4, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
scores = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
#test_data = open('test/test.csv').read().split('\n')
#tests = tokenizer.sequences_to_matrix(test_data, mode='binary')
predictions = model.predict(x_test, batch_size=128)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
rounded = [round(x[0]) for x in predictions]
print(x_test)
print(rounded)