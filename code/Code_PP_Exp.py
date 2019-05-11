# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:29:39 2019

@author: arijp
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
data = data.rename(index=str, columns={"zip.code": "zipcode", "house.color": "housecolor", "ni.age": "age", "len.at.res":"lenatres", "coverage.type": "covtype",
                                       "dwelling.type":"dwelltype", "sales.channel":"channel", "ni.gender":"gender", "ni.marital.status":"married", "n.adults":"nadults",
                                       "n.children":"nkids", "claim.ind":"claim"})

### Cleaning and Pre-processing ######

# 1) Look at plots of all variables to check for outliers
# 2) Missing values?
# 3) Any errors in data
# 4) What are continuous, categorical (ordinal, nominal), any temporal info?


# col names
names = list(data)

# number of columns with missing values
col_miss = len(data) - data.count()

# number of rows with at least one missing value
row_miss = data.shape[0] - data.dropna().shape[0]
row_clean = data.dropna().shape[0]

# plots per variables using Seaborn: note Seaborn does NOT include NAN values

# cancel ########################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x="cancel", data=data)
plt.show()

data['cancel'].value_counts()

    # Notes:
        # var file says only two levels, but there are three
        # what does value -1 stand for?
        
        # This variable is binary
        
#################################################################################

# year ##########################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x="year", data=data)
plt.show()

data['year'].value_counts()

    # Notes:
        # Is 2017 not included?
        
        # Continuous (positive, usually whole number)

#################################################################################

# zipcode ####################################################################### 

    # Notes: 
        # Has too many levels
        # Aggregate by state, county, region
        # Convert to lat-lon
        
        # Continuous

# housecolor ####################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x="housecolor", data=data)
plt.show()

data['housecolor'].value_counts()

    # Notes:
        # Contains missing
        
        # Categorical variable

#################################################################################

# age ###########################################################################
sns.set(style="darkgrid")
ax = sns.distplot(data['age'].dropna())
plt.show()

data['age'].value_counts()
    
    # Notes:
        # Some age values are not integers
        
        # Continuous variable

#################################################################################
        
# length at residence ###########################################################
sns.set(style="darkgrid")
ax = sns.distplot(data['lenatres'].dropna())
plt.show()

data['lenatres'].value_counts()

    # Notes:
        # Some lenatres values are not integers
        
        # Continuous variable
        
# credit ########################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x="credit", data=data)
plt.show()

data['credit'].value_counts()        

#################################################################################

# coverage type #################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x="covtype", data=data)
plt.show()

data['covtype'].value_counts()   

#################################################################################

# dwelling type #################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x="dwelltype", data=data)
plt.show()

data['dwelltype'].value_counts()   

#################################################################################

# Premium #######################################################################
sns.set(style="darkgrid")
ax = sns.distplot(data['premium'].dropna())
plt.show()

# Sales Channel #################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x = 'channel', data = data)
plt.show()

#################################################################################

# Gender ########################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x = 'gender', data = data)
plt.show()

#################################################################################

# Married #######################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x = 'married', data = data)
plt.show()

#################################################################################

# Number Adults #################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x = 'nadults', data = data)
plt.show()

#################################################################################

# Number Children ###############################################################
sns.set(style="darkgrid")
ax = sns.countplot(x = 'nkids', data = data)
plt.show()

#################################################################################

# Tenure ########################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x = 'tenure', data = data)
plt.show()

sns.set(style="darkgrid")
ax = sns.distplot(data['tenure'].dropna())
plt.show()

#################################################################################

# Claim #########################################################################
sns.set(style="darkgrid")
ax = sns.countplot(x = 'claim', data = data)
plt.show()

    # Notes:
        # Was the claim paid?

#################################################################################

###############################
######## Sliced Plots #########
###############################

sns.set(style="darkgrid")
ax = sns.countplot(x = 'cancel', hue='claim', data = data)
plt.show()

###############################
### Information from Zipcode ##
###############################

from uszipcode import SearchEngine
search = SearchEngine(simple_zipcode=True)

z = search.by_zipcode(data['zipcode'][0:1].astype(int))

###############################
### US Census Data ############
###############################
