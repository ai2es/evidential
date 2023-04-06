import numpy as np
import tensorflow.keras.backend as K


def monte_carlo_ensemble(model, x_test, y_test, forward_passes, y_scaler=None):
    """Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    y_scaler : sklearn Scaler
        perform inverse scaler on predicted
    """
    n_samples = x_test.shape[0]
    pred_size = y_test.shape[1]
    dropout_mu = np.zeros((forward_passes, n_samples, pred_size, 2))
    dropout_aleatoric = np.zeros((forward_passes, n_samples, pred_size))
    
    for i in range(forward_passes):
        output = model.model(x_test, training=True)
        mu, aleatoric  = model.calc_uncertainties(output.numpy(), y_scaler)
        dropout_mu[i] = np.dstack([mu, aleatoric])

    # Calculating mean across multiple MCD forward passes
    mean_mu = np.mean(dropout_mu, axis=0)  # shape (n_samples, n_classes)
    #mean_aleatoric = np.mean(dropout_aleatoric, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    variance_mu = np.var(dropout_mu, axis=0)  # shape (n_samples, n_classes)
    #variance_aleatoric = np.var(dropout_aleatoric, axis=0)  # shape (n_samples, n_classes)

    return mean_mu, variance_mu


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)