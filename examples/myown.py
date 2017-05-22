import keras
from preprocessor import Dataset, pad_vec_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, GRU, Activation
from keras.utils import np_utils, generic_utils
from keras.preprocessing.text import Tokenizer
import numpy as np

max_words = 1000
batch_size = 128
epochs = 100
ds = Dataset()
X_all = ds.X_all_sent
Y_all = ds.Y_all
x_train, x_test, y_train, y_test = train_test_split(
                                   X_all,Y_all,test_size=0.1, random_state=42)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
num_classes = 3 + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
scores = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
test_data = open('test/test.csv').read().split('\n')
tests = tokenizer.sequences_to_matrix(test_data, mode='binary')
predictions = model.predict(tests, batch_size=128)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
rounded = [round(x[0]) for x in predictions]
print(x_test)
print(predictions)