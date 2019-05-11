# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:07:36 2019

@author: arijp
"""

# Pre-process file

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import io
import requests
import math
from scipy import stats



imp_num = 1
# Import and rename data
data = pd.read_csv('imputed_'+str(imp_num)+'.csv')

# col names
names = list(data)
print(names)

#print (data.isnull().sum()) # same as above

# this is z-score that value minus mean divided by standard deviation
# http://duramecho.com/Misc/WhyMinusOneInSd.html
def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def str_to_int(df):
    str_columns = df.select_dtypes(['object']).columns
    print(str_columns)
    for col in str_columns:
        df[col] = df[col].astype('category')

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

def count_space_except_nan(x):
    if isinstance(x,str):
        return x.count(" ") + 1
    else :
        return 0
    
# https://stackoverflow.com/a/42523230
def one_hot(df, cols, drop = False, remove_orig = True):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=drop)
        if remove_orig:
            del df[each]
        df = pd.concat([df, dummies], axis=1)
    return df

def pre_processing(df,compute_zip = True):
    
    # Delete the "Area" column from the dataframe
    df = df.drop("Unnamed: 0", axis=1)
    
    # Remove ID
    #df.drop(['id'], axis=1, inplace=True)
    
    # Cancel
    df = df[df['cancel'] != -1] # remove rows with corrupted response
    df = one_hot(df, ['cancel'], drop = True)
    
    # Year
    # With so few years, maybe make it categorical: ordinal
    #df['year'] = df.replace({'year':{2013:0, 2014:1, 2015:2, 2016:3}})
    #df = one_hot(df,['year'], remove_orig = False)
    
    # House color
    df = one_hot(df, ['housecolor'])
    
    # Age
    # Since age refers to policyholder, we remove anyone over 110 years of age
    # Age Normalization
    
    # no need to normalize for xgboost
    # df['age'] = feature_normalize(df['age'])
    
    # Length at Residence
    # This doesn't seem to need any modification
    # Length at Residence Normalization
    # df['lenatres'] = feature_normalize(df['lenatres'])
    
    # Coverage Type
    # Two options here, make coverage type ordinal or nominal (lets keep it nominal)
    df = one_hot(df, ['covtype'])
    
    # Dwelling Type
    df = one_hot(df, ['dwelltype'])
    
    # Premium
    # This doesn't need changing yet (maybe consider normalizing)
    
    # no need to normalize for xgboost
    # df['premium'] = feature_normalize(df['premium'])
    
    # Channel
    df = one_hot(df, ['channel'])
    
    # Gender
    df = one_hot(df, ['gender'], drop=True)
    
    # Married
    df = one_hot(df, ['married'], drop=True)
    
    # Number of Adults
    
    # Number of Kids
    
    # Tenure
    
    # no need to normalize for xgboost
    # df['tenure'] = feature_normalize(df['tenure'])
    
    # Claim
    df = one_hot(df, ['claim'], drop=True)
    
    # State
    df = one_hot(df,['state'], remove_orig = False)
    
    # Credit
    ordered_credit = ['low', 'medium', 'high']
    df['credit_o'] = pd.DataFrame(df.credit.astype("category",ordered=True,categories=ordered_credit).cat.codes)   
    
    temp_credit = df['credit_o'].value_counts()
    temp_credit_prop = [i/np.sum(temp_credit) for i in temp_credit]
    
    # creating a dict file  
    cred_dict = {'low': temp_credit_prop[2],'medium': temp_credit_prop[1], 'high':temp_credit_prop[0]} 
    df.credit = [cred_dict[item] for item in df.credit]
    
    
    # Feature Engineering
    df['age_tenure'] = df['age']*df['tenure']
    
    df['nkids_married'] = df['nkids']*df['married_1']
    
    df['age_House'] = df['age']*df['dwelltype_House']
    
    return df

import time

start = time.time()
cleaned = pre_processing(data)
finish = time.time() - start
cleaned.to_csv('train_clean_pp_I'+str(imp_num)+'.csv', index=False)


'''
# After initial pre-processing check out interaction terms to see what might be of interest
# then rerun with any feature engineering

# boxplots
data.plot(kind='box', subplots=True, layout=(2,10), sharex=False, sharey=False)
from matplotlib import pyplot as plt
plt.show()

# basic scatter plot
from pandas.plotting import scatter_matrix
dat = data.drop(['Unnamed: 0','id','train','credit', 'state'], axis=1)
fig = scatter_matrix(dat.sample(n=10000))
plt.show()
plt.savefig('scatter2.png', bbox_inches='tight')


# Visual correlation
dat3 = dat
dat2 = dat3.sample(n=10000)
correlations = dat2.corr()
# plot correlation matrix
names = list(dat2)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,31,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
ax.set_xticklabels(names,rotation=90)
plt.show()
fig.savefig('corr2.png', bbox_inches='tight')

# Pull out top correlations
c = dat.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")

from matplotlib import pyplot as plt

plt.matshow(dat2.corr())
plt.show()

def get_redundant_pairs(df):
    #Get diagonal and lower triangular pairs of correlation matrix
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(dat3, 20))
'''

