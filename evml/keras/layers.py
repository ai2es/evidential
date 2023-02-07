import tensorflow as tf

class Evidential(tf.keras.layers.Layer):
    def __init__(self, classes=2):
        super(Evidential, self).__init__()
        self.classes = classes

    def call(self, inputs):
        evidence = K.softplus(inputs)
        alphas = evidence + 1
        S = tf.reduce_sum(alphas, axis=1, keepdims=True)
        u = self.classes / S
        prob = alphas / S
        belief = evidence / S
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob ** 2 - epistemic
        merged = tf.concat([belief, u, epistemic, aleatoric], 1)
        return merged