#!/usr/bin/env python
# coding: utf-8

# make the original data split into train and test, and make transformation
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def data_split(data, test_size = 0.1, random_state = 10):
    """
    Split data into train data and test data two parts by given proportion

    :param data: Original Data
    :param test_size: the proportion of cases in test data, default as 0.1
    :param random_state: Set the random seed to get the reproducible result, default as 10
    :return: train data and test data, each part has features X and response Y
    """
    data_tr, data_te = model_selection.train_test_split(data, test_size = test_size, random_state = random_state)

    # Drop the target from the training and testing data
    X_tr = data_tr.drop(columns = ['TARGET'])
    X_te = data_te.drop(columns = ['TARGET'])
    y_tr = data_tr['TARGET']
    y_te = data_te['TARGET']
    
    return X_tr, y_tr, X_te, y_te

# Second function
# input train X, test X and the starategy of imputing
# impute numeric missing value (default as the median) in the column
# then scale the features between 0 and 1
# return the imputed data

def data_impute(X_tr, X_te, starategy = 'median'): 
    """
    Impute numeric missing value in the column by the first data set
     
    :param X_tr: get imputing transformation by train data set
    :param X_te: also impute the test data set 
    :param starategy: imputing starategy, default as median in X_tr column
    :return: imputed train and test data
    """
    # Median imputation of missing values
    imputer = SimpleImputer(strategy = 'median')

    # Transform both training and testing data
    # Fit on the training data
    imputer.fit(X_tr)
    X_tr = imputer.transform(X_tr)
    X_te = imputer.transform(X_te)
    
    return X_tr, X_te

# Third function
# standardize data by the train data X
# input train and test data
# return standardized data

def data_standardize(X_tr, X_te):
    """
     Standardize X_tr and X_te data by X_tr
     
    :param X_tr: get standardized transformation by train data set
    :param X_te: also standardize the test data set 
    :return: standardized train and test data
    """
    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    
    return X_tr, X_te


