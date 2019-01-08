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



# Missing data
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# lets set our treshold at 1%

# Dealing with missing data
train_data = train_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
train_data.isnull().sum().max() # checking that there's no missing data missin




#standardizing data
saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)




# Bivariate analysis SalePrice/GrLivArea
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# Bivariate analysis SalePrice/TotalBsmtSF
var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# Deleting points
train_data.sort_values(by = 'GrLivArea', ascending = False)[:2]
train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
train_data = train_data.drop(train_data[train_data['Id'] == 524].index)

train_data.sort_values(by = 'TotalBsmtSF', ascending = False)[:4sh]
train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
train_data = train_data.drop(train_data[train_data['Id'] == 333].index)
train_data = train_data.drop(train_data[train_data['Id'] == 497].index)
train_data = train_data.drop(train_data[train_data['Id'] == 524].index)


# Shapiro-Wilk test for normality
import scipy
shapiro_results=scipy.stats.shapiro(train_data['SalePrice'])

matrix_sw = [
    ['', 'DF', 'Test Statistic', 'p-value'],
    ['Sample Data', len(train_data['SalePrice']) - 1, shapiro_results[0], shapiro_results[1]]
]



# Histogram and normal probability plot
sns.distplot(train_data['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)

# Applying log transformation
train_data['SalePrice'] = np.log(train_data['SalePrice'])

# Transformed histogram and normal probability plot
sns.distplot(train_data['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)


# Histogram and normal probability plot
train_data['GrLivArea'] = np.log(train_data['GrLivArea'])

sns.distplot(train_data['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_data['GrLivArea'], plot=plt)



# Histogram and normal probability plot
sns.distplot(train_data['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_data['TotalBsmtSF'], plot=plt)



# Let's create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
train_data['HasBsmt'] = pd.Series(len(train_data['TotalBsmtSF']), index=train_data.index)
train_data['HasBsmt'] = 0 
train_data.loc[train_data['TotalBsmtSF']>0,'HasBsmt'] = 1

# Then transform data
train_data.loc[train_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_data['TotalBsmtSF'])

# And again histogram and normal probability plot
sns.distplot(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# Homoscedasticity checkup

#scatter plot
plt.scatter(train_data['GrLivArea'], train_data['SalePrice']);


#scatter plot
plt.scatter(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], 
            train_data[train_data['TotalBsmtSF']>0]['SalePrice']);





# Convert categorical variable into dummy
train_data = pd.get_dummies(train_data)



print(train_v1.shape)

train_v2 = train_v2.select_dtypes(include=[np.number])
test_v2 = test_v2.select_dtypes(include=[np.number])

print(train_v2.shape)

## Skorzystać z tej strony 

#  https://www.kaggle.com/gabrielmlg/solution-house-prices




# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(train_data)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




