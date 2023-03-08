import tensorflow as tf


class DenseNormalGamma(tf.keras.layers.Layer):
    """Implements dense layer for Deep Evidential Regression
    
    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/aamini/evidential-deep-learning
    """
    
    def __init__(self, units, name, **kwargs):
        super(DenseNormalGamma, self).__init__(name=name, **kwargs)
        self.units = int(units)
        self.dense = tf.keras.layers.Dense(4 * self.units, activation=None)

    def evidence(self, x):
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 4 * self.units

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config
      