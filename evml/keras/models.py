import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.layers import Dense, LeakyReLU, GaussianNoise, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from evml.keras.layers import DenseNormalGamma, DenseNormal
from evml.keras.losses import EvidentialRegressionLoss, GaussianNLL
from evml.keras.losses import DirichletEvidentialLoss
from evml.keras.callbacks import ReportEpoch
from imblearn.under_sampling import RandomUnderSampler
from imblearn.tensorflow import balanced_batch_generator


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

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        evidential_coef=0.05,
        optimizer="adam",
        loss_weights=None,
        use_noise=False,
        noise_sd=0.01,
        uncertainties=True,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.1,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.01,
        l2_weight=0.01,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        verbose=0,
        save_path=".",
        model_name="model.h5",
        metrics=None,
    ):

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
        self.metrics = metrics

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.
        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """

        nn_input = Input(shape=(inputs.shape[1],), name="input")
        nn_model = nn_input

        if self.activation == "leaky":
            self.activation = LeakyReLU()

        if self.kernel_reg == "l1":
            self.kernel_reg = L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None

        for h in range(self.hidden_layers):
            nn_model = Dense(
                self.hidden_neurons,
                activation=self.activation,
                kernel_regularizer=L2(self.l2_weight),
                name=f"dense_{h:02d}",
            )(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(
                    nn_model
                )
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(
                    nn_model
                )
        nn_model = DenseNormalGamma(outputs.shape[-1], name="DenseNormalGamma")(
            nn_model
        )
        self.model = Model(nn_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr
            )  # , beta_1=self.adam_beta_1, beta_2=self.adam_beta_2)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        if self.metrics == "mae":
            metrics = self.mae
        elif self.metrics == "mse":
            metrics = self.mse
        else:
            metrics = None
        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=metrics,
            run_eagerly=False,
        )
        self.training_var = [np.var(outputs[:, i]) for i in range(outputs.shape[1])]

    def fit(
        self,
        x,
        y,
        validation_data=None,
        callbacks=None,
        initial_epoch=0,
        steps_per_epoch=None,
        workers=1,
        use_multiprocessing=False,
    ):
        # inputs = x.shape[1]
        # if len(y.shape) == 1:
        #     outputs = 1
        #     self.training_var = [np.var(y)]
        # else:
        #     outputs = y.shape[1]
        #     self.training_var = [np.var(y[:, i]) for i in range(y.shape[1])]
        self.build_neural_network(x, y)
        self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            callbacks=callbacks,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=True,
        )

        return

    def save_model(self):
        tf.keras.models.save_model(
            self.model, os.path.join(self.save_path, self.model_name), save_format="h5"
        )
        return

    def predict(self, x, scaler=None):
        y_out = self.model.predict(x, batch_size=self.batch_size)
        if self.uncertainties:
            y_out_final = self.calc_uncertainties(y_out, scaler)
        else:
            y_out_final = y_out
        return y_out_final

    def mae(self, y_true, y_pred):
        mu, _, _, _ = tf.split(y_pred, 4, axis=-1)
        return tf.keras.metrics.mean_absolute_error(y_true, mu)

    def mse(self, y_true, y_pred):
        mu, _, _, _ = tf.split(y_pred, 4, axis=-1)
        return tf.keras.metrics.mean_squared_error(y_true, mu)

    def calc_uncertainties(self, preds, y_scaler):
        mu, v, alpha, beta = np.split(preds, 4, axis=-1)
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))

        if mu.shape[-1] == 1:
            mu = np.expand_dims(mu, 1)
            aleatoric = np.expand_dims(aleatoric, 1)
            epistemic = np.expand_dims(epistemic, 1)

        if y_scaler:
            mu = y_scaler.inverse_transform(mu)

        for i in range(mu.shape[-1]):
            aleatoric[:, i] *= self.training_var[i]
            epistemic[:, i] *= self.training_var[i]

        return mu, aleatoric, epistemic
    
    def predict_dist_params(self, x):
        preds = self.model.predict(x, batch_size=self.batch_size)
        mu, v, alpha, beta = np.split(preds, 4, axis=-1)
        if mu.shape[-1] == 1:
            mu = np.expand_dims(mu, 1)
        if y_scaler:
            mu = y_scaler.inverse_transform(mu)
            
        return mu, v, alpha, beta


class ParametricRegressorDNN(EvidentialRegressorDNN):
    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.
        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        self.loss = GaussianNLL

        nn_input = Input(shape=(inputs.shape[1],), name="input")
        nn_model = nn_input

        if self.activation == "leaky":
            self.activation = LeakyReLU()

        if self.kernel_reg == "l1":
            self.kernel_reg = L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None

        for h in range(self.hidden_layers):
            nn_model = Dense(
                self.hidden_neurons,
                activation=self.activation,
                kernel_regularizer=L2(self.l2_weight),
                name=f"dense_{h:02d}",
            )(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(
                    nn_model
                )
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(
                    nn_model
                )
        nn_model = DenseNormal(outputs.shape[-1])(nn_model)
        self.model = Model(nn_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2
            )
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        if self.metrics == "mae":
            metrics = self.mae
        elif self.metrics == "mse":
            metrics = self.mse
        else:
            metrics = None
        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=metrics,
            run_eagerly=False,
        )
        self.training_var = [np.var(outputs[:, i]) for i in range(outputs.shape[1])]

    def mae(self, y_true, y_pred):
        mu, aleatoric = tf.split(y_pred, 2, axis=-1)
        return tf.keras.metrics.mean_absolute_error(y_true, mu)

    def mse(self, y_true, y_pred):
        mu, aleatoric = tf.split(y_pred, 2, axis=-1)
        return tf.keras.metrics.mean_squared_error(y_true, mu)

    def calc_uncertainties(self, preds, y_scaler):
        mu, aleatoric = np.split(preds, 2, axis=-1)
        if mu.shape[-1] == 1:
            mu = np.expand_dims(mu)
            aleatoric = np.expand_dims(aleatoric)
        if y_scaler:
            mu = y_scaler.inverse_transform(mu)
        for i in range(aleatoric.shape[-1]):
            aleatoric[:, i] *= self.training_var[i]
        return mu, aleatoric


class CategoricalDNN(object):

    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.
    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        output_activation: Activation function applied to the output layer
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss functions or loss objects (can match up to number of output layers)
        loss_weights: Weights to be assigned to respective loss/output layer
        use_noise: Whether or not additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        lr: Learning rate for optimizer
        use_dropout: Whether or not Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        l2_weight: L2 weight parameter
        sgd_momentum: SGD optimizer momentum parameter
        adam_beta_1: Adam optimizer beta_1 parameter
        adam_beta_2: Adam optimizer beta_2 parameter
        decay: Level of decay to apply to learning rate
        verbose: Level of detail to provide during training (0 = None, 1 = Minimal, 2 = All)
        classifier: (boolean) If training on classes
    """

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        output_activation="softmax",
        optimizer="adam",
        loss="categorical_crossentropy",
        loss_weights=None,
        annealing_coeff=None,
        use_noise=False,
        noise_sd=0.0,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.2,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.0,
        l2_weight=0.0,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        epsilon=1e-7,
        decay=0,
        verbose=0,
        classifier=False,
        random_state=1000,
        callbacks=[],
        balanced_classes=0,
        steps_per_epoch=0,
    ):

        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epsilon = epsilon
        self.loss = loss
        self.loss_weights = loss_weights
        self.annealing_coeff = annealing_coeff
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
        self.callbacks = callbacks
        self.decay = decay
        self.verbose = verbose
        self.classifier = classifier
        self.y_labels = None
        self.model = None
        self.random_state = random_state
        self.balanced_classes = balanced_classes
        self.steps_per_epoch = steps_per_epoch

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.
        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        if self.activation == "leaky":
            self.activation = LeakyReLU()

        if self.kernel_reg == "l1":
            self.kernel_reg = L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None

        self.model = tf.keras.models.Sequential()
        self.model.add(
            Dense(
                inputs,
                activation=self.activation,
                kernel_regularizer=self.kernel_reg,
                name="dense_input",
            )
        )

        for h in range(self.hidden_layers):
            self.model.add(
                Dense(
                    self.hidden_neurons,
                    activation=self.activation,
                    kernel_regularizer=self.kernel_reg,
                    name=f"dense_{h:02d}",
                )
            )
            if self.use_dropout:
                self.model.add(Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model.add(GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        self.model.add(
            Dense(outputs, activation=self.output_activation, name="dense_output")
        )

        if self.optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
                epsilon=self.epsilon
            )
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(
                learning_rate=self.lr, momentum=self.sgd_momentum
            )

        self.model.build((self.batch_size, inputs))
        self.model.compile(optimizer=self.optimizer_obj, loss=self.loss)

    def fit(self, x_train, y_train, validation_data=None):

        inputs = x_train.shape[-1]
        outputs = y_train.shape[-1]

        if self.loss == "dirichlet":
            for callback in self.callbacks:
                if isinstance(callback, ReportEpoch):
                    # Don't use weights within Dirichelt, it is done below using sample weight
                    self.loss = DirichletEvidentialLoss(
                        callback=callback, name=self.loss
                    )
                    break
            else:
                raise OSError(
                    "The ReportEpoch callback needs to be used in order to run the evidential model."
                )
        self.build_neural_network(inputs, outputs)
        if self.balanced_classes:
            train_idx = np.argmax(y_train, 1)
            training_generator, steps_per_epoch = balanced_batch_generator(
                x_train,
                y_train,
                sample_weight=np.array([self.loss_weights[_] for _ in train_idx]),
                sampler=RandomUnderSampler(),
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
            history = self.model.fit(
                training_generator,
                validation_data=validation_data,
                steps_per_epoch=steps_per_epoch,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                shuffle=True,
            )
        else:
            sample_weight = np.array([self.loss_weights[np.argmax(_)] for _ in y_train])
            if not self.steps_per_epoch:
                self.steps_per_epoch = sample_weight.shape[0] // self.batch_size
            history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                sample_weight=sample_weight,
                steps_per_epoch=self.steps_per_epoch,
                #class_weight={k: v for k, v in enumerate(self.loss_weights)},
                shuffle=True,
            )
        return history

    def predict(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size, verbose=self.verbose)
        return y_prob

    def predict_dropout(self, x, mc_forward_passes=10):
        y_prob = np.stack(
            [
                np.vstack(
                    [
                        self.model(tf.expand_dims(lx, axis=-1), training=True)
                        for lx in np.array_split(x, x.shape[0] // self.batch_size)
                    ]
                )
                for _ in range(mc_forward_passes)
            ]
        )
        pred_probs = y_prob.mean(axis=0)
        epistemic_variance = y_prob.var(axis=0)
        # Calculating entropy across multiple MCD forward passes
        epsilon = sys.float_info.min
        entropy = -np.sum(
            pred_probs * np.log(pred_probs + epsilon), axis=-1
        )  # shape (n_samples,)
        # Calculating mutual information across multiple MCD forward passes
        mutual_info = entropy - np.mean(
            np.sum(-y_prob * np.log(y_prob + epsilon), axis=-1), axis=0
        )  # shape (n_samples,)
        return pred_probs, epistemic_variance, entropy, mutual_info

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size, verbose=self.verbose)
        return y_prob
    
    def compute_uncertainties(self, y_pred, num_classes = 4):
        return calc_prob_uncertainty(y_pred, num_classes = num_classes)
    
    
    
def calc_prob_uncertainty(y_pred, num_classes = 4):
    evidence = tf.nn.relu(y_pred)
    alpha = evidence + 1
    S = tf.keras.backend.sum(alpha, axis=1, keepdims=True)
    u = num_classes / S
    prob = alpha / S
    epistemic = prob * (1 - prob) / (S + 1)
    aleatoric = prob - prob**2 - epistemic
    return prob, u, aleatoric, epistemic