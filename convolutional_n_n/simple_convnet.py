import sys, os
# imac용 디렉토리 패스
sys.path.append('/Users/skdm/Documents/GitHub/deeplearning_python')
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class SimpleConvNet:
    