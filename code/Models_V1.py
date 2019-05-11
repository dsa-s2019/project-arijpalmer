# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:09:23 2019

@author: arijp
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import io
import requests
import math
from scipy import stats

import xgboost as xgb
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

from sklearn import cross_validation

# Import and rename data
data = pd.read_csv('train_clean_pp_I1.csv')

data['year'].values[data['year'] == 2017] = 2016
#,'age','lenatres','tenure'
#data['kar'] = data.nkids/data.nadults
#data['d'] = data.tenure/data.lenatres
#data['d2'] = data.age/data.tenure
#data['d3'] = data.age/data.lenatres
data_train = data[data.train == 1]
data_train = data_train.drop(['state','train','id','credit_o'], axis=1)

data_test = data[data.train == 0]
data_test_id = data_test['id']
data_test = data_test.drop(['state','train','id','cancel_1.0','credit_o'], axis=1)


y_dat = data_train['cancel_1.0']
X_dat = data_train.drop(['cancel_1.0'], axis=1)

X_train, X_test, y_train,y_test = cross_validation.train_test_split(X_dat, y_dat, test_size=0.10)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
d_oos = xgb.DMatrix(data_test)

param = {'max_depth':8, 'eta':.02, 'objective': 'binary:logistic', 'scale_pos_weight':3, 'subsample':0.7}
param['nthread'] = 4

param['eval_metric'] = ['error', 'auc']

evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 500
bst = xgb.train(param, dtrain, num_round, evallist)

ypred = bst.predict(dtest)

xgb.plot_importance(bst)
plt.show()
#plt.savefig('imp_feats.png', bbox_inches='tight')

subid = np.asarray(data_test_id)
writeout = pd.DataFrame(np.hstack((subid.reshape(-1,1),ypred.reshape(-1,1))))
#writeout.to_csv('preds1_newvars_year_long.csv', index=False)


import scikitplot as skplt
skplt.metrics.plot_roc(y_test, ypred)
plt.show()

import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, ypred)
roc_auc = metrics.auc(fpr, tpr)

fpr2, tpr2, threshold = metrics.roc_curve(y_test, ypred)
roc_auc2 = metrics.auc(fpr2, tpr2)

fpr3, tpr3, threshold = metrics.roc_curve(y_test, ypred)
roc_auc3 = metrics.auc(fpr3, tpr3)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC1 = %0.8f' % roc_auc)
plt.plot(fpr2, tpr2, 'k', label = 'AUC2 = %0.8f' % roc_auc2)
plt.plot(fpr3, tpr3, 'g', label = 'AUC3 = %0.8f' % roc_auc3)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig('auc2.png', bbox_inches='tight')

'''
#load saved model
import pickle

bst.save_model('xgb_m1.model')
imported_model2 = xgb.Booster(model_file='xgb_m1.model')

pred2 = imported_model2.predict(dtest)
'''