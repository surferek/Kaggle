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

# Choosen variables
"""
Neighborhood
OverallQual
YearBuilt
TotalBsmtSF
GrLivArea
"""

# Sale price statistics
train_data["SalePrice"].describe()

#histogram
sns.distplot(train_data['SalePrice'])

#skewness and kurtosis
print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())


numeric = ['TotalBsmtSF','GrLivArea']

categorical = ['Neighborhood','OverallQual','YearBuilt']

#scatter plots of numeric variables
for i in numeric:
    data = pd.concat([train_data['SalePrice'], train_data[i]], axis=1)
    data.plot.scatter(x=i, y='SalePrice', ylim=(0,800000));



#scatter plots of categorical variables
for i in categorical:
    data = pd.concat([train_data['SalePrice'], train_data[i]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=i, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000);



# Create correlation matrix
corrmx = train_data.corr()
f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmx, vmax=0.85,square = True)



# Sale price correlation matrix for top 10 variables

k = 10 # Number of variables for our heatmap
cols = corrmx.nlargest(k,"SalePrice")["SalePrice"].index

cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale= 1.2)
heatmp = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = ' .2f',
                     annot_kws={'size':9},
                     yticklabels= cols.values, xticklabels= cols.values)
plt.show()


# Scatterplot for choosen variables

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 
        'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[cols], size = 2.5)
plt.show();










