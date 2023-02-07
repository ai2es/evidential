import tensorflow as tf 
import numpy as np
from functools import partial


class DenseNormalGamma(tf.keras.layers.Layer):
    """Implements dense layer for Deep Evidential Regression
    
    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/aamini/evidential-deep-learning
    """
    
    def __init__(self, units):
        super(DenseNormalGamma, self).__init__()
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
        return (input_shape[0], 4 * self.units)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config
      
class EvidentialRegressorDNN(object):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.
    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        inputs: Number of input values
        outputs: Number of output values
        activation: Type of activation function
        output_activation: Activation function applied to the output layer
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object
        use_noise: Whether or not additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        use_dropout: Whether or not Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
    """
    def __init__(self, hidden_layers=1, hidden_neurons=4, activation="relu", evidential_coef=0.05,
                 optimizer="adam", loss_weights=None, use_noise=False, noise_sd=0.01, uncertainties=True,
                 lr=0.001, use_dropout=False, dropout_alpha=0.1, batch_size=128, epochs=2, kernel_reg='l2',
                 l1_weight=0.01, l2_weight=0.01, sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999, decay=0, 
                 verbose=0, training_std=None):
        
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.evidential_coef = evidential_coef
        self.loss = partial(self.EvidentialRegression, coeff=self.evidential_coef)
        self.uncertainties = uncertainties
        self.loss_weights = loss_weights
        self.lr = lr
        self.kernel_reg = kernel_reg
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.decay = decay
        self.verbose = verbose
        self.model = None
        self.optimizer_obj = None
        self.training_std = None

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.
        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        
        nn_input = Input(shape=(inputs,), name="input")
        nn_model = nn_input
        
        if self.activation == 'leaky':
            self.activation = LeakyReLU()
        
        if self.kernel_reg == 'l1':
            self.kernel_reg = l1(self.l1_weight)
        elif self.kernel_reg == 'l2':
            self.kernel_reg = l2(self.l2_weight)
        elif self.kernel_reg == 'l1_l2':
            self.kernel_reg = l1_l2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None
        
        for h in range(self.hidden_layers):
            nn_model = Dense(self.hidden_neurons, activation=self.activation,
                             kernel_regularizer=l2(self.l2_weight), name=f"dense_{h:02d}")(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(nn_model)
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(nn_model)
        nn_model = DenseNormalGamma(outputs)(nn_model)
        self.model = Model(nn_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2, decay=self.decay)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum, decay=self.decay)
        self.model.compile(optimizer=self.optimizer_obj, loss=self.loss, loss_weights=self.loss_weights, run_eagerly=False)

    def fit(self, x, y):
        inputs = x.shape[1]
        if len(y.shape) == 1:
            outputs = 1
        else:
            outputs = y.shape[1]
        self.build_neural_network(inputs, outputs)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, shuffle=True)
        self.training_var = np.var(y).values
        return

    def predict(self, x, scaler):
        
        y_out = self.model.predict(x, batch_size=self.batch_size)
        if self.uncertainties:
            return self.calc_uncertainties(y_out, scaler)
        else:
            return y_out


    def NIG_NLL(self, y, gamma, v, alpha, beta, reduce=True):
        # v += 1e-12
        twoBlambda = 2*beta*(1+v)# + 1e-12
        nll = 0.5*tf.math.log(np.pi/v)  \
            - alpha*tf.math.log(twoBlambda)  \
            + (alpha+0.5) * tf.math.log(v*(y-gamma)**2 + twoBlambda)  \
            + tf.math.lgamma(alpha)  \
            - tf.math.lgamma(alpha+0.5)

        return tf.reduce_mean(nll) if reduce else nll

    def NIG_Reg(self, y, gamma, v, alpha, beta, reduce=True):
        error = tf.abs(y-gamma)
        evi = 2*v+(alpha)
        reg = error*evi

        return tf.reduce_mean(reg) if reduce else reg

    def EvidentialRegression(self, y_true, evidential_output, coeff=1.0):
        """Implements loss for Deep Evidential Regression

        Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
        Source: https://github.com/aamini/evidential-deep-learning
        """

        gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
        loss_nll = self.NIG_NLL(y_true, gamma, v, alpha, beta)
        loss_reg = self.NIG_Reg(y_true, gamma, v, alpha, beta)
        return loss_nll + coeff * loss_reg
    
    def calc_uncertainties(self, preds, y_scaler):
        mu, v, alpha, beta = (preds[:, i] for i in range(preds.shape[1]))
        mu = y_scaler.inverse_transform(mu.reshape((mu.shape[0], -1))).squeeze()
        aleatoric = np.sqrt((beta / (alpha - 1)) * self.training_var)
        epistemic = np.sqrt((beta / (v * (alpha - 1))) * self.training_var)
        return np.array([mu, aleatoric, epistemic]).T

      
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

def calc_prob_uncertainty(outputs, num_classes = 4):
    evidence = tf.nn.relu(outputs)
    alpha = evidence + 1
    S = tf.keras.backend.sum(alpha, axis=1, keepdims=True)
    u = num_classes / S
    prob = alpha / S
    epistemic = prob * (1 - prob) / (S + 1)
    aleatoric = prob - prob**2 - epistemic
    return prob, u, aleatoric, epistemic