from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
from typing import List, Dict
import logging
from functools import partial
import math


logger = logging.getLogger(__name__)


def get_callbacks(config: Dict[str, str]) -> List[Callback]:

    callbacks = []

    if "callbacks" in config:
        config = config["callbacks"]
    else:
        return []

    if "ModelCheckpoint" in config:
        callbacks.append(ModelCheckpoint(**config["ModelCheckpoint"]))
        logger.info("... loaded Checkpointer")

    if "EarlyStopping" in config:
        callbacks.append(EarlyStopping(**config["EarlyStopping"]))
        logger.info("... loaded EarlyStopping")

    # LearningRateTracker(),  ## ReduceLROnPlateau does this already, use when supplying custom LR annealer

    if "ReduceLROnPlateau" in config:
        callbacks.append(ReduceLROnPlateau(**config["ReduceLROnPlateau"]))
        logger.info("... loaded ReduceLROnPlateau")

    if "CSVLogger" in config:
        callbacks.append(CSVLogger(**config["CSVLogger"]))
        logger.info("... loaded CSVLogger")
        
    if "LearningRateScheduler" in config:
        drop = config["LearningRateScheduler"]["drop"]
        epochs_drop = config["LearningRateScheduler"]["epochs_drop"]
        f = partial(step_decay, drop = drop, epochs_drop = epochs_drop)
        callbacks.append(LearningRateScheduler(f))
        callbacks.append(LearningRateTracker())

    return callbacks


def step_decay(epoch, drop=0.2, epochs_drop=5.0, init_lr=0.001):
    lrate = init_lr * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
    return lrate


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = {}) -> None:
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class ReportEpoch(tf.keras.callbacks.Callback):
    def __init__(self, annealing_coeff):
        super(ReportEpoch, self).__init__()
        self.this_epoch = 0
        self.annealing_coeff = annealing_coeff
    
    def on_epoch_begin(self, epoch, logs={}):
        self.this_epoch += 1


class CSICallback(tf.keras.callbacks.Callback):
    def __init__(self, model, x_test, y_test):
        super(PredictCallback, self).__init__()
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.x_test))
        logs["val_csi"] = self.mean_csi(self.y_test, y_pred)
        return
    

    def mean_csi(self, y_true, y_pred):
        pred_probs, u = calc_prob_uncertinty(y_pred)
        pred_probs = pred_probs.numpy()
        u = u.numpy()
        #true_labels = np.argmax(y_true, 1) 
        pred_labels = np.argmax(pred_probs, 1)
        confidences = np.take_along_axis(
            pred_probs,
            pred_labels[:, None], 
            axis=1
        )
        rocs = []
        for i in range(pred_probs.shape[1]):
            forecasts = confidences.copy()
            obs = np.where(np.argmax(y_true, 1)  == i, 1, 0)
            roc = DistributedROC(thresholds=np.arange(0.0, 1.01, 0.01),
                                 obs_threshold=0.5)
            roc.update(forecasts[:, 0], obs)
            rocs.append(roc.max_csi())
        return np.mean(rocs)
        
class ClassifierMetrics(tf.keras.callbacks.Callback):
    
    def __init__(self, x_valid, y_valid, x_test, y_test, n_bins = 15, **kwargs):
        
        super(ClassifierMetrics, self).__init__(**kwargs)
        self.X_valid = x_valid
        self.Y_valid = y_valid
        self.X_test = x_test
        self.Y_test = y_test
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        y_valid_pred = np.asarray(self.model.predict(self.X_valid))
        y_test_pred = np.asarray(self.model.predict(self.X_test))

        y_valid_labels = np.argmax(self.Y_valid, axis=1)
        y_test_labels = np.argmax(self.Y_test, axis=1)
        #y_valid_pred_labels = np.argmax(y_valid_pred, axis=1)
        #y_test_pred_labels = np.argmax(y_test_pred, axis=1)
        
        logs = {
            'val_ave_acc': self.ave_acc(y_valid_labels, y_valid_pred),
            'val_mce': self.mce(y_valid_labels, y_valid_pred),
            'val_ece': self.ece(y_valid_labels, y_valid_pred),
            'test_ave_acc': self.ave_acc(y_test_labels, y_test_pred),
            'test_mce': self.mce(y_test_labels, y_test_pred),
            'test_val_ece': self.ece(y_test_labels, y_test_pred)
        }
        self._data.append(logs)
        
        return

    def get_data(self):
        return self._data
    
    
    def ave_acc(self, true_labels, pred_probs):
        accs = []
        pred_labels = np.expand_dims(np.argmax(pred_probs, 1), -1)
        return np.mean(
            [
                (
                    true_labels[np.where(true_labels == _label)]
                    == pred_labels[np.where(true_labels == _label)]
                ).mean()
                for _label in np.unique(true_labels)
            ]
        )

    
    def mce(self, true_labels, pred_probs):
        confidences = np.expand_dims(np.max(pred_probs, 1), -1)
        predictions = np.expand_dims(np.argmax(pred_probs, 1), -1)
        accuracies = predictions == true_labels

        mce = 0.0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower).astype(float) * (
                confidences <= bin_upper
            ).astype(float)
            prop_in_bin = in_bin.astype(float).mean()
            in_bin = in_bin.squeeze(-1).astype(int)
            if prop_in_bin > 0:
                try:
                    max_accuracy_in_bin = accuracies[in_bin].astype(float).max()
                    max_confidence_in_bin = confidences[in_bin].max()
                    max_calibration = np.abs(max_confidence_in_bin - max_accuracy_in_bin)
                    mce = max(mce, max_calibration)
                except:
                    pass
        return mce

    
    def ece(self, true_labels, pred_probs):
        confidences = np.expand_dims(np.max(pred_probs, 1), -1)
        predictions = np.expand_dims(np.argmax(pred_probs, 1), -1)
        accuracies = predictions == true_labels
        ece = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower).astype(float) * (
                confidences <= bin_upper
            ).astype(float)
            prop_in_bin = in_bin.astype(float).mean()
            in_bin = in_bin.squeeze(-1).astype(int)
            if prop_in_bin > 0:
                try:
                    accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    avg_calibration = (
                        np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    )
                    ece.append(avg_calibration)
                except:
                    pass
        return np.mean(ece)