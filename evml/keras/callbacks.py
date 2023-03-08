import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class ReportEpoch(tf.keras.callbacks.Callback):
    def __init__(self, annealing_coef):
        super(ReportEpoch, self).__init__()
        self.this_epoch = 0
        self.annealing_coef = annealing_coef
    
    def on_epoch_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.this_epoch += 1
