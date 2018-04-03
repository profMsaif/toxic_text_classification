import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

print('####################################################################')
print('                 classifying online toxic comments                  ')
print('####################################################################')
print('choose on of the following option to continue : ')
print('1- classify  dataset of comments using logestic regression')
print('2- classify  single comment using logestic regression')
print('3- split data and find the accuracity  ')
print('4- classify  dataset of comments  using keras ')
print('5- load a pretrain keras model ')


print('--------------------------------------------------------------------')


#fig,ax = plt.subplots(2,3,figsize=(16,10))
#ax1,ax2,ax3,ax4,ax5,ax6 = ax.flatten()
#sns.countplot(train['toxic'],palette= 'magma',ax=ax1)
#sns.countplot(train['severe_toxic'], palette= 'viridis',ax=ax2)
#sns.countplot(train['obscene'], palette= 'Set1',ax=ax3)
#sns.countplot(train['threat'], palette= 'viridis',ax = ax4)
#sns.countplot(train['insult'], palette = 'magma',ax=ax5)
#sns.countplot(train['identity_hate'], palette = 'Set1', ax = ax6)
#plt.show()


#------------------------------------------------------------------------------------------
def split_data(x_train_data,y_train_data):
	print('--------------------------------------------------------------------')
	size_test_data=float(input("choose the size of the test data (0.33 recommended): "))
	x_train, x_test, y_train, y_test = train_test_split(x_train_data,y_train_data, test_size=size_test_data, random_state=42)
	return(x_train, x_test, y_train, y_test)
	
	
#-------------------------------------------------------------------------------------------
#X_train and X_test must be a datafrmae
def vectorizing_data_set_for_LR(X_train,X_test): 
	print('--------------------------------------------------------------------')                                         
	print("Vectorizering the comments in x_train and x_test")
	vect = CountVectorizer()
	if what_to_do==2:
		X_train.loc[len(X_train)]=X_test
		vect.fit(X_train)
		X_train_vect=vect.transform(X_train.iloc[0:len(X_train)-1])
		X_test_vect=vect.transform([X_train.iloc[len(X_train)-1]])
	elif what_to_do==1 or what_to_do==3:		
		all_data=X_train.append(X_test)
		vect.fit(all_data)
		X_train_vect=vect.transform(all_data.iloc[0:len(X_train)])
		X_test_vect=vect.transform(all_data.iloc[len(X_train):])
	return(X_train_vect,X_test_vect)
#--------------------------------------------------------------------------------------------
#preparing x_trian x_test and y_train
def prepare_data():
	print('--------------------------------------------------------------------')
	train_data_path=input("train data path is (data/train.csv) :  ")
	print('--------------------------------------------------------------------')
	train_data = pd.read_csv(train_data_path).fillna(' ')
	class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	X_train_data=train_data["comment_text"]
	Y_train_data=train_data[class_names]
	if what_to_do == 3 :
		print("the size of the train data is ",train_data.shape)
		x_train, x_test, y_train, y_test=split_data(X_train_data,Y_train_data)
	elif what_to_do == 1 :	
		test_data_path=input("test data path is (data/test.csv) :  ")
		test_data= pd.read_csv(test_data_path).fillna(' ')
		x_train=X_train_data
		y_train=Y_train_data
		x_test=test_data["comment_text"]
		y_test=pd.DataFrame(columns=class_names)
	elif what_to_do == 2 :
		x_train=X_train_data
		y_train=Y_train_data
		x_test=input("write commentt to classify:  ")
		print('--------------------------------------------------------------------')
		y_test=pd.DataFrame(columns=class_names)
	x_train_vectorized,x_test_vectorized=vectorizing_data_set_for_LR(x_train,x_test)
	print('--------------------------------------------------------------------')
	return(x_train_vectorized,x_test_vectorized,y_train,y_test,class_names,x_test)
#-------------------------------------------------------------------------------------------
#logistic regression
def classifying_with_log_reg(x_train_vectorized,x_test_vectorized,y_train,y_test,class_names):
	model = LogisticRegression()
	if what_to_do == 1 or what_to_do==2 :
		print("what type of prediction ? ")
		print("Enter 1 for probability prediction , 2 for true/false prediction")
		while True:	
			z=int(input("Your Entry is  = "))
			if z==1 or z==2 :
				break
			else:
				print("you have to choose either 1 or 2")
		print("please wait , this will take some time")
		for i in range (0,len(class_names)):
			y_train_temp=y_train[str(class_names[i])]	
			model.fit(x_train_vectorized,y_train_temp)
			if z ==1 :
				pred=model.predict_proba(x_test_vectorized)
				y_test[str(class_names[i])]=pred[:,1]
			elif z==2 :	
				pred=model.predict(x_test_vectorized)
				y_test[str(class_names[i])]=pred
			else:
				print("wrong entry please!")
			print("train LR model with x_train and ", str(class_names[i])," columns")
		y_pred=y_test
		print("-----------------------classification is done!----------------------")
		
	elif what_to_do == 3 :
		print("train data to find accuracy")
		print('--------------------------------------------------------------------')
		print("please wait , this will take some time")
		print('--------------------------------------------------------------------')
		y_pred=pd.DataFrame(columns=class_names)
		for i in range (0,len(class_names)):
			y_train_temp=y_train[str(class_names[i])]	
			model.fit(x_train_vectorized,y_train_temp)
			pred=model.predict(x_test_vectorized)
			y_pred[str(class_names[i])]=pred
			print("train LR model with x_train and ", str(class_names[i])," columns")
		print("-----------------------classification is done!----------------------")
	return(y_pred)
#-------------------------------------------------------------------------------------------------
def toknizing(X):
	from keras.preprocessing.text import Tokenizer
	print("extracting features and fixing short sentences")
	max_features = 20000
	tokenizer = Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(X))
	tokenized_X = tokenizer.texts_to_sequences(X)
	#maxumim length	
	maxlen = 200
	X_tok= pad_sequences(tokenized_X, maxlen=maxlen)	
	return(X_tok)
#--------------------------------------------------------------------------------------------------------
def classifier_with_keras():	
	print('--------------------------------------------------------------------')
	print("loading data ........................................................")
	#print('--------------------------------------------------------------------')
	#train_data_path=input("train data path is :  ")
	train_data = pd.read_csv('data/train.csv').fillna(' ')
	#print('--------------------------------------------------------------------')
	#test_data_path=input("test data path is :  ")
	#print('--------------------------------------------------------------------')
	#test_data = pd.read_csv('data/test.csv').fillna(' ')
	class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	#define X and Y for  training set as an array
	X_train_start=train_data["comment_text"].values
	Y_train= train_data[class_names]
	X_train=toknizing(X_train_start)
	h=int(input("choose model 1- my model or 2-LSTM model: "))	
	if h==1:
		print("creating and training the model")
		model = Sequential()
		model.add(Dense(64, activation='relu', input_shape=(200,)))
		model.add(Dropout(0.5))
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(6, activation='softmax'))
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		model.fit(X_train, Y_train,epochs=10,batch_size=128)
		model.summary()
	elif h==2 :
		print("creating and training the model")
		#identifing the input
		inp = Input(shape=(200, ))
		embed_size = 128
		x = Embedding(20000, embed_size)(inp)
		x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
		x = GlobalMaxPool1D()(x)
		x = Dropout(0.1)(x)
		x = Dense(50, activation="relu")(x)
		x = Dropout(0.1)(x)
		x = Dense(6, activation="sigmoid")(x)
		model = Model(inputs=inp, outputs=x)
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		batch_size = 32
		epochs = 2
		model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs)
		model.summary()
		from keras import backend as K
		get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[2].output])
		layer_output = get_3rd_layer_output([X_train[:1]])[0]
		layer_output.shape
		#print layer_output to see the actual data
		#prediction = model.predict(np.array(tk.texts_to_sequences(text)))
		#p
		# save model to single file
	model.save(input("save as: ")+'.h5')
	

#-------------------------------------------------------------------------------------------------------
def load_keras_and_predict():
	model_loaded = load_model(input("load keras model (file path): ")+'.h5')
	#x_to_predict=input("input comment to predict:")
	#x_to_predict=[x_to_predict]
	#max_features = 20000
	#tokenizer = Tokenizer(num_words=max_features)
	#tokenizer.fit_on_texts(list(x_to_predict))
	#tokenized_x_to_predict = tokenizer.texts_to_sequences(x_to_predict)
	#maxlen = 200
	#x_tokenized= pad_sequences(tokenized_x_to_predict, maxlen=maxlen)
	#predicition=model_loaded.predict_classes(x_tokenized)
	test_data = pd.read_csv('data/test.csv').fillna(' ')
	class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	X_test_start=test_data["comment_text"].values
	X_test=toknizing(X_test_start)
	new_pred=model_loaded.predict(X_test)
	print(new_pred)
	return(new_pred)
#--------------------------------------------------------------------------------------------------------
while True:
	what_to_do=int(input("what you want to do (choose 1,2,3,4,5) : "))
	if what_to_do ==1:
		x_train_vectorized,x_test_vectorized,y_train,y_test,class_names,x_test=prepare_data()
		y_prediction=classifying_with_log_reg(x_train_vectorized,x_test_vectorized,y_train,y_test,class_names)
		result_final=pd.concat([x_test,y_prediction],axis=1)
		result_file_name=input("saving result as csv file (write the name --WITHOUT-- the extention): ")
		result_final.to_csv(result_file_name+'.csv')
					
	elif what_to_do==2:
		x_train_vectorized,x_test_vectorized,y_train,y_test,class_names,x_test=prepare_data()
		y_prediction=classifying_with_log_reg(x_train_vectorized,x_test_vectorized,y_train,y_test,class_names)
		print('the result for the following comment"',x_test,'"')
		print(y_prediction)
	elif what_to_do==3:
		x_train_vectorized,x_test_vectorized,y_train,y_test,class_names,x_test=prepare_data()
		y_prediction=classifying_with_log_reg(x_train_vectorized,x_test_vectorized,y_train,y_test,class_names)
		print("the accuracy score = ",accuracy_score(y_test, y_prediction))
	elif what_to_do==4:
		from keras.preprocessing.text import Tokenizer
		from keras.preprocessing.sequence import pad_sequences
		from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
		from keras.layers import Bidirectional, GlobalMaxPool1D
		from keras.models import Model, Sequential
		from keras import initializers, regularizers, constraints, optimizers, layers
		classifier_with_keras()
	elif what_to_do==5:
		from keras.preprocessing.text import Tokenizer
		from keras.preprocessing.sequence import pad_sequences
		from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
		from keras.layers import Bidirectional, GlobalMaxPool1D
		from keras.models import Model, Sequential ,load_model
		from keras import initializers, regularizers, constraints, optimizers, layers
		the_prediction=load_keras_and_predict()
	elif what_to_do==0:
		break
	else: 
		print("you have to choose one of option on the list  , to exit  type 0")

	
#-------------------------------------------------------------------------------------



