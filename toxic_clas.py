#import librires 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import sparse
seed = 42
import os
os.environ['OMP_NUM_THREADS'] = '4'

print('[importing data]')
#import data
train_data=pd.read_csv('data/train.csv')
test_data=pd.read_csv('data/test.csv')
print('data was succsusfally imoprted into train_data,test_data')
#all coulmns name
#columns_name=train_data.columns.tolist()
print('[inistilazing data]')
#initilaize data
class_names=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
comment = 'comment_text'
train_data_x=train_data[comment]
train_data_y=train_data.iloc[:,2:8]
test_data_x=test_data[comment]
print(['train_data was split into train_data_x and train_data_y'])

print('[exploreing data]')
fig,ax = plt.subplots(2,3,figsize=(16,10))
ax1,ax2,ax3,ax4,ax5,ax6 = ax.flatten()
sns.countplot(train_data['toxic'],palette= 'magma',ax=ax1)
sns.countplot(train_data['severe_toxic'], palette= 'viridis',ax=ax2)
sns.countplot(train_data['obscene'], palette= 'Set1',ax=ax3)
sns.countplot(train_data['threat'], palette= 'viridis',ax = ax4)
sns.countplot(train_data['insult'], palette = 'magma',ax=ax5)
sns.countplot(train_data['identity_hate'], palette = 'Set1', ax = ax6)
print('[show the plots]')
plt.show()

#--------------------
missing_values = pd.DataFrame()
missing_values['train_data'] = train_data.isnull().sum()
missing_values['test_data'] = test_data.isnull().sum()
missing_values
test_data.fillna(' ',inplace=True)
gc.collect()
#Text preprossesing
#source Term Frequency Inverse Document Frequency Vectorizer
print('[ text preprossesing ]')
vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word',
                        stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
vect_char = TfidfVectorizer(max_features=40000, lowercase=True, analyzer='char',
                        stop_words= 'english',ngram_range=(3,6),dtype=np.float32)


tr_vect = vect_word.fit_transform(train_data['comment_text'])
ts_vect = vect_word.transform(test_data['comment_text'])
print('[continue ...]')
# Character n gram vector
tr_vect_char = vect_char.fit_transform(train_data['comment_text'])
ts_vect_char = vect_char.transform(test_data['comment_text'])
gc.collect()

#--------------------
X = sparse.hstack([tr_vect, tr_vect_char])
x_test = sparse.hstack([ts_vect, ts_vect_char])

target_col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[target_col]
del tr_vect, ts_vect, tr_vect_char, ts_vect_char
gc.collect()

prd = np.zeros((x_test.shape[0],y.shape[1]))
cv_score =[]
for i,col in enumerate(target_col):
    lr = LogisticRegression(C=2,random_state = i,class_weight = 'balanced')
    print('Building {} model for column:{''}'.format(i,col)) 
    lr.fit(X,y[col])
    #cv_score.append(lr.score)
    prd[:,i] = lr.predict_proba(x_test)[:,1]
#Model Validation on train data set
ol = 'identity_hate'
print("Column:",col)
pred =  lr.predict(X)
print('\nConfusion matrix\n',confusion_matrix(y[col],pred))
print(classification_report(y[col],pred))

#Roc AUC curve
col = 'identity_hate'
print("Column:",col)
pred_pro = lr.predict_proba(X)[:,1]
frp,trp,thres = roc_curve(y[col],pred_pro)
auc_val =auc(frp,trp)
plt.figure(figsize=(14,10))
plt.plot([0,1],[0,1],color='b')
plt.plot(frp,trp,color='r',label= 'AUC = %.2f'%auc_val)
plt.legend(loc='lower right')
plt.xlabel('True positive rate')
plt.ylabel('False positive rate')
plt.title('Reciever Operating Characteristic')
plt.show()

