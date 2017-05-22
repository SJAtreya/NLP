from keras.models import Sequential
from keras.layers import Dense
import numpy
from preprocessor import Dataset, pad_vec_sequences
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, GRU, Activation
from keras.utils import np_utils, generic_utils

hidden_dim=32
print("Preparing Dataset..")
ds = Dataset()
print("Padding vector sequences..")
X_all = pad_vec_sequences(ds.X_all_vec_seq)
Y_all = ds.Y_all
batch_size=128
num_epoch=100
num_classes=ds.num_classes
print("Splitting training and test data")
x_train, x_test, y_train, y_test = train_test_split(
                                   X_all,Y_all,test_size=0.1, random_state=42)


print('X_train shape:', x_train.shape)
# create model
model = Sequential()
model.add(LSTM(128, input_shape=x_train.shape[1:]))
model.add(Dense(1))
#model.add(Dense(12, activation='relu'))
model.add(Activation('sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(x_train, y_train, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(x_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)