# import tensorflow_hub as hub
import tensorflow as tf
import os
import numpy as np
import copy
import pickle
import math
import torch
from disa_util import *
tf.compat.v1.disable_eager_execution()

# model = filter_model()
# model = filter_model_hgd()
model = gpt2_model()
tf_config = tf.ConfigProto(allow_soft_placement=True)        
sess = tf.Session(config=tf_config)
saver = tf.train.Saver()
model_ckpt = 'gpt2/model.ckpt-100000'
saver.restore(sess,model_ckpt)
model.test_model(sess)






