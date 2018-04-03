#import all librires 
#librires to manipulate dat from csv file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

import seaborn as sns

train = pd.read_csv('data/train.csv').fillna(' ')
test = pd.read_csv('data/test.csv').fillna(' ')
#train.columns.tolist()
#test.columns.tolist()

#identifing class names
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

fig,ax = plt.subplots(2,3,figsize=(16,10))
ax1,ax2,ax3,ax4,ax5,ax6 = ax.flatten()
sns.countplot(train['toxic'],palette= 'magma',ax=ax1)
sns.countplot(train['severe_toxic'], palette= 'viridis',ax=ax2)
sns.countplot(train['obscene'], palette= 'Set1',ax=ax3)
sns.countplot(train['threat'], palette= 'viridis',ax = ax4)
sns.countplot(train['insult'], palette = 'magma',ax=ax5)
sns.countplot(train['identity_hate'], palette = 'Set1', ax = ax6)
plt.show()

#define X and Y for  training set
X_train=train["comment_text"].values
X_test=test["comment_text"].values
Y_train = train[class_names]
print('[step3 is done]')
print('################################################')
print('[step4:             extracting features        ]')
print('################################################')
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
print('[step4 is done]')
print('################################################')
print('[step5 :         fixing the short sentences    ]')
print('################################################')
maxlen = 200
X_train_final = pad_sequences(tokenized_X_train, maxlen=maxlen)
X_test_final = pad_sequences(tokenized_X_test, maxlen=maxlen)
print('[step5 is done]')
print('################################################')
print('[    1-classifying using logistic regression   ]')
print('################################################')
logisticRegr_model = LogisticRegression()
Y_test=pd.DataFrame(columns=class_names)
for i in range(0,6):
	logisticRegr_model.fit(X_train_final,Y_train[str(class_names[i])].values)
	pred_temp=logisticRegr_model.predict_proba(X_test_final)
	result_one_class=pd.DataFrame(pred_temp)
	Y_test[str(class_names[i])]=result_one_class[0]
	print("Done! ",i+1," iteration")
print("Done! looping")
result_final=pd.concat([test,Y_test],axis=1)
result_final.to_csv('result_keras.csv')
#score = logisticRegr_model.score(X_test_final, Y_test)
#print(score)


X_test=test["comment_text"].values
X_test[0]	
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_test[0]))
tokenized_X_test0 = tokenizer.texts_to_sequences(X_test[0])
maxlen = 200
X_test_final = pad_sequences(tokenized_X_test0, maxlen=maxlen)











