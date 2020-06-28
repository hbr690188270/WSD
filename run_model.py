# import tensorflow_hub as hub
import tensorflow as tf
import os
import numpy as np
import copy
import pickle
import math
import torch
from disa_util import *

# model = filter_model()
# model = filter_model_hgd()
model = bert_filter_2()
model.test_model(filter = None)

# model = dense_filter()
# model.test_model()


# model = SAT_filter()
# model.test_model()


# model = cip15_filter()
# model.test_model()

# model = bert_filter_new()
# model.test_model(filter = None)
