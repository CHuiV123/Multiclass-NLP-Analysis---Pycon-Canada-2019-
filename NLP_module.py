#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:48:01 2022

@author: angela
"""
#%%

from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional 
from tensorflow.keras import Sequential, Input
import matplotlib.pyplot as plt

#%%
class ModelDevelopment:
    def LSTM_model(self, input_shape,vocab_size,out_dim,nb_node=128,dropout_rate=0.3):
        '''
        

        Parameters
        ----------
        input_shape : TYPE
            DESCRIPTION: input shape is the data shape of X train
        nb_class : TYPE
            DESCRIPTION: nb_class is the total of class we have in this dataset
        nb_node : TYPE, optional
            DESCRIPTION. The default is 128.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.3.
        activation : TYPE, optional
            DESCRIPTION. 

        Returns
        -------
        None.

        '''
        model = Sequential()
        model.add(Input(shape=(input_shape)))
        model.add(Embedding(vocab_size,out_dim))
        model.add(LSTM(nb_node,return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(nb_node)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(5, activation='softmax'))
        model.summary()

        return model


class ModelEvaluation():
    def LOSS_plot(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.show()
    
    def ACC_plot(self,hist): 
        plt.figure()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.show()

    
    