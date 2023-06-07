import tensorflow as tf
import numpy as np


class DirichletEvidentialLoss(tf.keras.losses.Loss):
    def __init__(self, callback=False, name="dirichlet"):

        super().__init__()
        self.callback = callback
        self.__name__ = name

    def KL(self, alpha):
        beta = tf.constant(np.ones((1, alpha.shape[1])), dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(
            tf.math.lgamma(alpha), axis=1, keepdims=True
        )
        lnB_uni = tf.reduce_sum(
            tf.math.lgamma(beta), axis=1, keepdims=True
        ) - tf.math.lgamma(S_beta)

        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)

        kl = (
            tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True)
            + lnB
            + lnB_uni
        )
        return kl

    def __call__(self, y, output, sample_weight = None):
        evidence = tf.nn.relu(output)
        alpha = evidence + 1

        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        m = alpha / S

        A = tf.reduce_sum((y - m) ** 2, axis=1, keepdims=True)
        B = tf.reduce_sum(
            alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True
        )

        annealing_coef = tf.minimum(
            1.0, self.callback.this_epoch / self.callback.annealing_coef
        )
        alpha_hat = y + (1 - y) * alpha
        C = annealing_coef * self.KL(alpha_hat)
        C = tf.reduce_mean(C, axis=1)
        return tf.reduce_mean(A + B + C)


#     def get_config(self):
#         base_config = {}
#         base_config['callback'] = self.callback
#         base_config['weights'] = self.weights
#         base_config['name'] = self.__name__
#         return base_config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


class EvidentialRegressionLoss(tf.keras.losses.Loss):
    def __init__(self, coeff=1.0):
        super(EvidentialRegressionLoss, self).__init__()
        self.coeff = coeff

    def NIG_NLL(self, y, gamma, v, alpha, beta, reduce=True):
        v = tf.math.maximum(v, tf.keras.backend.epsilon())
        twoBlambda = 2 * beta * (1 + v)
        nll = (
            0.5 * tf.math.log(np.pi / v)
            - alpha * tf.math.log(twoBlambda)
            + (alpha + 0.5) * tf.math.log(v * (y - gamma) ** 2 + twoBlambda)
            + tf.math.lgamma(alpha)
            - tf.math.lgamma(alpha + 0.5)
        )

        return tf.reduce_mean(nll) if reduce else nll

    def NIG_Reg(self, y, gamma, v, alpha, reduce=True):
        error = tf.abs(y - gamma)
        evi = 2 * v + alpha
        reg = error * evi

        return tf.reduce_mean(reg) if reduce else reg

    def call(self, y_true, evidential_output):
        gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
        loss_nll = self.NIG_NLL(y_true, gamma, v, alpha, beta)
        loss_reg = self.NIG_Reg(y_true, gamma, v, alpha)

        return loss_nll + self.coeff * loss_reg

    def get_config(self):
        config = super(EvidentialRegressionLoss, self).get_config()
        config.update({"coeff": self.coeff})
        return config


def GaussianNLL(y, y_pred, reduce=True):
    ax = list(range(1, len(y.shape)))
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
    loss = tf.reduce_mean(-logprob, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss


class DirichletInformedPriorLoss(tf.keras.losses.Loss):
    def __init__(self, callback=False, name="dirichletIP"):

        super().__init__()
        self.callback = callback
        self.__name__ = name

    def KL(self, alpha, prior_dist):
        beta = prior_dist
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        
        A = tf.math.lgamma(S_alpha) - tf.math.lgamma(S_beta) #ln Gamma(K) 
        
        B = (
            tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) 
             - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
            )

        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)

        kl = (
            tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True)
            + A
            + B
        )
        return kl

    def __call__(self, y, output, sample_weight = None):
        '''
        note: need to ensure that the informed prior sums to a known constant (e.g. 1 or K) in order
        to derive uncertainty value at inference time
        '''
        
        evidence = tf.nn.relu(output)
        
        y_len = int(y.shape[1] / 2)
        prior_dist = y[: , y_len: ] #2nd half of y
        y = y[: ,  :y_len] #first half of y

        alpha = evidence + prior_dist

        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        m = alpha / S

        A = tf.reduce_sum((y - m) ** 2, axis=1, keepdims=True)
        B = tf.reduce_sum(
            alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True
        )

        annealing_coef = tf.minimum(
            1.0, self.callback.this_epoch / self.callback.annealing_coef
        )
        alpha_hat = y + (1 - y) * alpha
        C = annealing_coef * self.KL(alpha_hat, prior_dist)
        C = tf.reduce_mean(C, axis=1)
        return tf.reduce_mean(A + B + C)

class EvidentialRegressionFixLoss(tf.keras.losses.Loss):
    def __init__(self, r=1.0, coeff=1.0, regularize=False):
        '''
        implementation of the loss from meinert and lavin that fixes issues with the original
        evidential loss for regression. The loss couples the virtual evidence values with coefficient r. 
        In this new loss, the regularizer is unneccessary.
        '''
        super(EvidentialRegressionFixLoss, self).__init__()
        self.coeff = coeff
        self.regularize = regularize #meinert and lavin claim the regularizer is unecessary with the new loss
        self.r = r

    def NIG_NLL(self, y, gamma, v, alpha, beta, reduce=True):
        v = tf.math.maximum(v, tf.keras.backend.epsilon()) #may need to delete this line?
        #couple the parameters as per meinert and lavin
        alpha = 0.5 * self.r * v #can also couple the other way with v = ..
        
        twoBlambda = 2 * beta * (1 + v)
        nll = (
            0.5 * tf.math.log(np.pi / v)
            - alpha * tf.math.log(twoBlambda)
            + (alpha + 0.5) * tf.math.log(v * (y - gamma) ** 2 + twoBlambda)
            + tf.math.lgamma(alpha)
            - tf.math.lgamma(alpha + 0.5)
        )

        return tf.reduce_mean(nll) if reduce else nll

    def NIG_Reg(self, y, gamma, v, alpha, reduce=True):
        alpha = 0.5 * self.r * v # coupling from meinert and lavin
        
        error = tf.abs(y - gamma) # can try squared loss here to target the right minimizer
        evi = 2 * v + alpha #new paper: = v + 2 * alpha
        reg = error * evi

        return tf.reduce_mean(reg) if reduce else reg

    def call(self, y_true, evidential_output):
        gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)

        loss_nll = self.NIG_NLL(y_true, gamma, v, alpha, beta)
        loss_reg = self.NIG_Reg(y_true, gamma, v, alpha)

        return loss_nll + self.regularize * self.coeff * loss_reg

    def get_config(self):
        config = super(EvidentialRegressionFixLoss, self).get_config()
        config.update({"r": self.r, "coeff": self.coeff, "regularize": self.regularize})
        return config