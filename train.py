#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, warnings, sys
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import mlflow


# define column names to be used
COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country']

CONTINUOUS_COLUMNS = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                      'capital-loss', 'hours-per-week']

CATEGORICAL_COLUMNS = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country']

TARGET = 'target'


# main part to run the code
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(42)

    if len(sys.argv) < 3:
        raise ValueError('Train and Test data path are required parameters!')

    # parse train and test data path
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    # parse LigthGBM parameters
    num_boost_round = 1000
    learning_rate = 0.1
    num_leaves = 31
    max_depth = -1
    min_data_in_leaf = 20

    if len(sys.argv) > 3:
        num_boost_round = int(sys.argv[3])

    if len(sys.argv) > 4:
        learning_rate = float(sys.argv[4])

    if len(sys.argv) > 5:
        num_leaves = int(sys.argv[5])

    if len(sys.argv) > 6:
        max_depth = int(sys.argv[6])

    if len(sys.argv) > 7:
        min_data_in_leaf = int(sys.argv[7])

    # read dataset
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # pre-process the numerical variables
    for column in CONTINUOUS_COLUMNS:
        median = train[column].median()
        train[column] = train[column].fillna(value=median)
        test[column] = test[column].fillna(value=median)

    # pre-process the categorical variables
    for column in CATEGORICAL_COLUMNS:
        train[column] = train[column].fillna(value='Missing')
        test[column] = test[column].fillna(value='Missing')

    # pre-process the target value
    train[TARGET] = (train[TARGET] == '>50K').astype(int)
    test[TARGET] = (test[TARGET] == '>50K').astype(int)

    # encode categorical variable into numerical format
    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        train[column] = encoder.fit_transform(train[column])
        test[column] = encoder.transform(test[column])

    # define LightGBM paremeters
    params = {'objective': 'binary',
              'boosting': 'gbdt',
              'is_unbalance': True,
              'metric': ['auc'],
              'learning_rate': learning_rate,
              'num_leaves': num_leaves,
              'max_depth': max_depth,
              'min_data_in_leaf': min_data_in_leaf,
              'bagging_fraction': 1.0,
              'feature_fraction': 1.0,
              'bagging_freq': 0,
              'lambda_l1': 0.0,
              'lambda_l2': 0.0,
              'drop_rate': 0.1,
              'seed': 42,
              'verbosity': -1}

    # create LigthGBM dataset
    train_x = train[COLUMNS]
    train_y = train[TARGET].values

    gbm_train = lgb.Dataset(data=train_x, label=train_y, feature_name=COLUMNS,
                            categorical_feature=CATEGORICAL_COLUMNS,
                            free_raw_data=False)

    # 5-folder cross validation
    history = lgb.cv(params=params, train_set=gbm_train, nfold=5,
                     num_boost_round=num_boost_round, stratified=True,
                     early_stopping_rounds=10, verbose_eval=False)

    # get result
    cv_auc = history['auc-mean'][-1]
    best_rounds = len(history['auc-mean'])

    # re-train the model with best rounds
    model = lgb.train(params=params, train_set=gbm_train, num_boost_round=best_rounds)
    pred = model.predict(test[COLUMNS])

    # calculate the performance
    fpr, tpr, _ = roc_curve(test[TARGET].values, pred)
    test_auc = np.round(auc(fpr, tpr), 5)

    # output the logs
    print('LigthGBM Boosting Model:\n')
    print('{0:20s}{1:6.4f}'.format('learning_rate', learning_rate))
    print('{0:20s}{1:6d}'.format('num_leaves', num_leaves))
    print('{0:20s}{1:6d}'.format('max_depth', max_depth))
    print('{0:20s}{1:6d}'.format('min_data_in_leaf', min_data_in_leaf))
    print('{0:20s}{1:6d}'.format('best_rounds', best_rounds))
    print('{0:20s}{1:6.4f}'.format('cv_auc', cv_auc))
    print('{0:20s}{1:6.4f}'.format('test_auc', test_auc))

    # mlflow logs
    mlflow.log_param('max_rounds', num_boost_round)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('num_leaves', num_leaves)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('min_data_in_leaf', min_data_in_leaf)
    mlflow.log_param('best_rounds', best_rounds)

    mlflow.log_metric('cv_auc', cv_auc)
    mlflow.log_metric('test_auc', test_auc)
