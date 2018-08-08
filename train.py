#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, warnings, sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import mlflow


# define column name to be used
COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country')

CONTINUOUS_COLUMNS = ('age', 'fnlwgt', 'education-num', 'capital-gain',
                      'capital-loss', 'hours-per-week')

CAGETORICAL_COLUMNS = ('workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country')

TARGET = 'target'


# main part to run the code
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(42)

    # parse parameters from command line
