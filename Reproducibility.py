import numpy as np
import tensorflow as tf
import random
import os

seed = 121
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
