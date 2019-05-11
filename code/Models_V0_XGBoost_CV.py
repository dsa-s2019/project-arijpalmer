# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:09:23 2019

@author: arijp
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime

from xgboost import XGBClassifier
from sklearn import cross_validation

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# Import and rename data
data = pd.read_csv('train_clean_pp_I1.csv')
data = data[data.train == 1]
#data['kar'] = data.nkids/data.nadults

y_dat = data['cancel_1.0']
X_dat = data.drop(['id','train','credit','state','cancel_1.0'], axis=1)

X_train, X_test, y_train,y_test = cross_validation.train_test_split(X_dat, y_dat, test_size=0.2, random_state=0)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

# The data is huge, so maybe don't need so many cv
folds = 3

# Test Enough of these
param_comb = 1

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

# early stopping to decrease time
xgb = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic', silent=True, nthread=1)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=8, cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_train, y_train)
timer(start_time) # timing ends here for "start_time" variable

results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)