
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
print('####################################################################')
print('                classifying online toxic comments                   ')
print('####################################################################')
print('                  loading data ....................                 ')
print('[importing data]')
#import data
train_data=pd.read_csv('data/train.csv')
#test_data=pd.read_csv('data/test.csv')
test_data=pd.read_csv('result2.csv')
print('')
print('data was succsusfally imoprted into train_data,test_data')

train_data.isnull().any()
test_data.isnull().any()
print('[initialze data]')
classes_name = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#print('the classes are :',classes_name)
#creating an array of the output , contain two values 1,0 for each class
# 1 means the comment belong to this class .

print('')
y = train_data[classes_name].values
train = train_data["comment_text"]
test = test_data["comment_text"]
y_test = test_data[classes_name].values
print('[extract features : break down the sentence into unique words]')
# we cextract feature from the text and convert it to numbers
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train))
tokenized_train = tokenizer.texts_to_sequences(train)
tokenized_test = tokenizer.texts_to_sequences(test)
print('')
print('[fixing the short sentences]')
maxlen = 200
X_train = pad_sequences(tokenized_train, maxlen=maxlen)
X_test = pad_sequences(tokenized_test, maxlen=maxlen)

#print('[the distribution of the number of words in sentences.]')
#totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]

#plt.hist(totalNumWords,bins = np.arange(0,410,10))
#plt.show()


inp = Input(shape=(maxlen, ))


embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(6, activation="sigmoid")(x)


model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

batch_size = 32
epochs = 3
model.fit(X_train,y, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

model.summary()

from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_3rd_layer_output([X_train[:1]])[0]
layer_output.shape
#print layer_output to see the actual data


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(200,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train, y,epochs=2,batch_size=32,validation_split=0.33)

#score = model.evaluate(X_test, y_test, batch_size=128)

# For custom metrics


