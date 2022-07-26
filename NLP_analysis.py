#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:25:22 2022

@author: angela
"""
#%% IMPORTS

from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.preprocessing.text import Tokenizer 
from sklearn.model_selection import train_test_split
from NLP_module import ModelDevelopment,ModelEvaluation
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Sequential
from tensorflow.keras.utils import plot_model 

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import datetime
import pickle
import json
import re
import os 

#%% CONSTANTS 

LOGS_PATH = os.path.join(os.getcwd(),
                         'logs',
                         datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(),'models','tokenizer.json')

OHE_SAVE_PATH = os.path.join(os.getcwd(),'models','ohe.pkl')

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'models','model.h5')

#%% 1) Data Loading 

df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')


#%% 2) Data Inspection

df.head()
df.duplicated().sum()

df['text'].duplicated().sum()
df['category'].duplicated().sum()
print(df['text'][5]) 

#we will need to remove space

# Symbols and HTML tags have to be remove 

#%% 3) Data cleaning 

# to drop duplicate
df1 = df.drop_duplicates() #> decide to not remove duplicates after this operation produce imbalance dataset for later train test split purposes 
df1.duplicated().sum()


review = df['text']
category = df['category']

review_backup = review.copy()
category_backup = category.copy()

for index, word in enumerate(review):
    # to remove html tags
    # Anything within the <> will be removed including <> 
    #? to tell re dont be greedy so it wont capture everything 
    # from the first < to the last > in the document 
    review[index] = re.sub('<.*?>','',word)
    review[index] = re.sub('[^a-zA-Z]',' ',word).lower().split()
    review[index] = re.sub('[0-9]','',word)

#%% 4) Features Selection 

# there is no feature to select here

#%% 5) Data preprocessing 

# at this stage, we need to make sure there is no any empty space is our data
# use tokenizer, one hot encoding


vocab_size = 10000 # 1/5 of data size 
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review)  # to learn 
word_index = tokenizer.word_index

print(dict(list(word_index.items())[0:10]))  # to slice the data, show only 10
print(dict(list(word_index.items())))   # to show all 

review_int = tokenizer.texts_to_sequences(review) # To convert into numbers 
review_int[100]


length_review = []
for i in range(len(review_int)): 
    length_review.append(len(review_int[i]))
    # print(len(review_int[i]))

np.median(length_review)

max_len = np.median([len(review_int[i])for i in range(len(review_int))])


padded_review=pad_sequences(review_int,
                            maxlen=int(max_len),
                            padding='post',
                            truncating='post')


#Y target

ohe=OneHotEncoder(sparse=False)
category=ohe.fit_transform(np.expand_dims(category,axis=-1))



X_train,X_test,y_train,y_test=train_test_split(padded_review,category,
                                               test_size=0.3,
                                               random_state=(123))


#%% model development 

input_shape = np.shape(X_train)[1:]
out_dim = 128 
vocab_size = 10000

md = ModelDevelopment()
model = md.LSTM_model(input_shape,vocab_size,out_dim)

plot_model(model, show_shapes=(True))

model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics=['acc'])


#%% Model Training  

tensorboard_callback=TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)


hist = model.fit(X_train, y_train, 
                 epochs=6,
                 validation_data=(X_test, y_test),
                 callbacks=[tensorboard_callback])

# 6 epoch here is yielding averagely better accuracy 


#%% model evaluation 

print(hist.history.keys())

me = ModelEvaluation()
me.LOSS_plot(hist)
me.ACC_plot(hist)

print(model.evaluate(X_test,y_test)) #model score

#%% model analysis 


y_pred = np.argmax(model.predict(X_test),axis=1)
y_actual = np.argmax(y_test, axis=1)

labels = ['tech','business','sport','entertainment','politics']
cr = classification_report(y_actual, y_pred, target_names = labels)

print(cr)


cm = confusion_matrix(y_actual, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm ,display_labels =labels)
disp.plot(cmap='Oranges')
plt.rcParams['figure.figsize'] = [7, 7]
plt.show()

#%% model saving 

# TOKENIZER 
token_json = tokenizer.to_json()
with open(TOKENIZER_SAVE_PATH,'w') as file: 
    json.dump(token_json,file)


# OHE 
with open(OHE_SAVE_PATH,'wb') as file: 
    pickle.dump(ohe,file)

# MODEL
model.save(MODEL_SAVE_PATH)
