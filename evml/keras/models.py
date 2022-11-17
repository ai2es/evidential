import tensorflow as tf 
from keras import backend as K
import numpy as np

def build_model(input_size, 
                hidden_size, 
                num_hidden_layers, 
                output_size, 
                activation, 
                dropout_rate, 
                out_activation):
    
    model = tf.keras.models.Sequential()
        
    if activation == 'leaky':
        model.add(tf.keras.layers.Dense(input_size))
        model.add(tf.keras.layers.LeakyReLU())
        
        for i in range(num_hidden_layers):
            if num_hidden_layers == 1:
                model.add(tf.keras.layers.Dense(hidden_size))
                model.add(tf.keras.layers.LeakyReLU())
            else:
                model.add(tf.keras.layers.Dense(hidden_size))
                model.add(tf.keras.layers.LeakyReLU())
                model.add(tf.keras.layers.Dropout(dropout_rate))
    else:
        model.add(tf.keras.layers.Dense(input_size, activation=activation))
        
        for i in range(num_hidden_layers):
            if num_hidden_layers == 1:
                model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
            else:
                model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
                model.add(tf.keras.layers.Dropout(dropout_rate))
      
    model.add(tf.keras.layers.Dense(output_size, activation=out_activation))
    
    return model


def calc_prob_uncertinty(outputs, num_classes = 4):
    evidence = K.relu(outputs)
    alpha = evidence + 1
    u = num_classes / K.sum(alpha, axis=1, keepdims=True)
    prob = alpha / K.sum(alpha, axis=1, keepdims=True)
    return prob, u