import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
import numpy as np
from typing import List, Dict
import logging
from functools import partial
import math
from .models import calc_prob_uncertainty

logger = logging.getLogger(__name__)


class ReportEpoch(tf.keras.callbacks.Callback):
    def __init__(self, annealing_coeff):
        super(ReportEpoch, self).__init__()
        self.this_epoch = 0
        self.annealing_coeff = annealing_coeff
    
    def on_epoch_begin(self, epoch, logs={}):
        self.this_epoch += 1