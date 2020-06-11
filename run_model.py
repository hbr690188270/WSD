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
model = bert_filter()
model.test_model("cluster")






