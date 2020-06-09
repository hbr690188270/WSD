import sys
import os
import argparse
import json
import re

import tensorflow as tf
import numpy as np

from train.modeling import GroverModel, GroverConfig, sample
from tokenization import tokenization





