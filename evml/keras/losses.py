import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def Dirichlet(y, output, callback = False, weights = False):
    
    def KL(alpha):
        beta=tf.constant(np.ones((1, alpha.shape[1])), dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)

        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)

        if weights is not False:
            kl = tf.reduce_sum(weights * (alpha - beta)*(dg1-dg0), axis=1, keepdims=True) + lnB + lnB_uni
        else:
            kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
        return kl
    
    evidence = K.relu(output)
    alpha = evidence + 1
    
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    m = alpha / S

    if weights is not False:
        A = tf.reduce_sum(weights * (y-m)**2, axis=1, keepdims=True)
        B = tf.reduce_sum(weights * alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)
    else:
        A = tf.reduce_sum((y-m)**2, axis=1, keepdims=True)
        B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)

    annealing_coef = tf.minimum(1.0, callback.this_epoch/callback.annealing_coeff)
    alpha_hat = y + (1-y)*alpha
    C = annealing_coef * KL(alpha_hat)
    C = tf.reduce_mean(C, axis=1)
    return tf.reduce_mean(A + B + C)