import random
import os
import numpy as np
import tensorflow as tf

def seed_everything(seed=1234):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
