import tensorflow as tf


class DenseNormalGamma(tf.keras.layers.Layer):
    """Implements dense layer for Deep Evidential Regression

    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/aamini/evidential-deep-learning
    """

    def __init__(self, units, name, eps=1e-7, **kwargs):
        # 1e-7 is smallest we can go for float32
        super(DenseNormalGamma, self).__init__(name=name, **kwargs)
        self.eps = eps
        self.units = int(units)
        self.dense = tf.keras.layers.Dense(4 * self.units, activation=None)

    def evidence(self, x):
        return tf.math.maximum(tf.nn.softplus(x), self.eps)

    def call(self, x):
        output = self.dense(
            x
        )  # for float 64s change output = tf.cast(self.dense(x), tf.float64)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        # add 1 to get alpha
        alpha = tf.add(
            self.evidence(logalpha), tf.convert_to_tensor(1.0, tf.float32)
        )  # change this line for float64s
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 4 * self.units

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config["units"] = self.units
        return base_config


class DenseNormal(tf.keras.layers.Layer):
    def __init__(self, units):
        super(DenseNormal, self).__init__()
        self.units = int(units)
        self.dense = tf.keras.layers.Dense(
            2 * self.units, activation="sigmoid", eps=1e-12
        )
        self.eps = eps

    def call(self, x):
        output = self.dense(x)
        output = tf.math.maximum(output, self.eps)
        mu, sigma = tf.split(output, 2, axis=-1)
        # mu = tf.nn.sigmoid(mu) #+ tf.keras.backend.epsilon()
        # sigma = tf.nn.softplus(logsigma) + tf.keras.backend.epsilon()
        return tf.concat([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.units)

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config["units"] = self.units
        return base_config
