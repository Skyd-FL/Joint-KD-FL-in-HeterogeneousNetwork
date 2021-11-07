# from-root
import sys
from from_root import from_root
sys.path.append(str(from_root()))
# dataset
from emnist import list_datasets
from emnist import extract_training_samples

# complement
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import random
import time
import os
from scipy.special import softmax
import yaml
from datetime import datetime

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D

# sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split