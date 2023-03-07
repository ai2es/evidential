import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.layers import Dense, LeakyReLU, GaussianNoise, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from evml.keras.layers import DenseNormalGamma
from evml.keras.losses import EvidentialRegressionLoss


class EvidentialRegressorDNN(object):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.
    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        evidential_coef: Evidential regularization coefficient
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object
        use_noise: Whether additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        use_dropout: Whether Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
    """
    def __init__(self, hidden_layers=1, hidden_neurons=4, activation="relu", evidential_coef=0.05,
                 optimizer="adam", loss_weights=None, use_noise=False, noise_sd=0.01, uncertainties=True,
                 lr=0.001, use_dropout=False, dropout_alpha=0.1, batch_size=128, epochs=2, kernel_reg='l2',
                 l1_weight=0.01, l2_weight=0.01, sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999,
                 verbose=0, save_path='.',  model_name='model.h5'):
        
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.evidential_coef = evidential_coef
        self.loss = EvidentialRegressionLoss(coeff=self.evidential_coef)
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
        self.verbose = verbose
        self.save_path = save_path
        self.model_name = model_name
        self.model = None
        self.optimizer_obj = None
        self.training_std = None
        self.training_var = None

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
            self.kernel_reg = L1(self.l1_weight)
        elif self.kernel_reg == 'l2':
            self.kernel_reg = L2(self.l2_weight)
        elif self.kernel_reg == 'l1_l2':
            self.kernel_reg = L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None
        
        for h in range(self.hidden_layers):
            nn_model = Dense(self.hidden_neurons, activation=self.activation,
                             kernel_regularizer=L2(self.l2_weight), name=f"dense_{h:02d}")(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(nn_model)
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(nn_model)
        nn_model = DenseNormalGamma(outputs, name='DenseNormalGamma')(nn_model)
        self.model = Model(nn_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum)
        self.model.compile(optimizer=self.optimizer_obj, loss=self.loss,
                           loss_weights=self.loss_weights, run_eagerly=False)

    def fit(self, x, y):
        inputs = x.shape[1]
        if len(y.shape) == 1:
            outputs = 1
        else:
            outputs = y.shape[1]
        self.build_neural_network(inputs, outputs)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, shuffle=True)
        self.training_var = np.var(y)
        return

    def save_model(self):
        tf.keras.models.save_model(self.model,
                                   os.path.join(self.save_path,
                                                self.model_name),
                                   save_format='h5')
        return
    def predict(self, x, scaler=None):
        
        y_out = self.model.predict(x, batch_size=self.batch_size)
        if self.uncertainties:
            y_out_final = self.calc_uncertainties(y_out, scaler)
        else:
            y_out_final = y_out
        return y_out_final

    def calc_uncertainties(self, preds, y_scaler):
        mu, v, alpha, beta = (preds[:, i] for i in range(preds.shape[1]))
        if y_scaler:
            mu = y_scaler.inverse_transform(mu.reshape((mu.shape[0], -1))).squeeze()
        else:
            mu = mu.reshape((mu.shape[0], -1)).squeeze()
        aleatoric = np.sqrt((beta / (alpha - 1)) * self.training_var)
        epistemic = np.sqrt((beta / (v * (alpha - 1))) * self.training_var)
        return np.array([mu, aleatoric, epistemic]).T
