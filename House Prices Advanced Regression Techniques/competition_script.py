# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:24:15 2018

@author: Piotr
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


# Importing training and test data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Check for column names
train_data.columns
































