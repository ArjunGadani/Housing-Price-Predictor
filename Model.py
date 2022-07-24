# HOUSING PRICE PREDICTOR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


housing = pd.read_csv('data/Housing_Data.csv')
print(housing.head())

print(housing.info())

print(housing.describe())

housing.hist(bins = 50, figsize = (20, 15))
plt.scatter(housing['CHAS'], housing['MEDV'])

# Train Test spliting

# Statified Shuffle Split

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 1)

for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]    
    strat_test_set = housing.loc[test_index]
    
print(strat_test_set['CHAS'].value_counts())

housing_train = strat_train_set.copy()
housing_test = strat_test_set.copy()
print(housing_train.shape)
print(housing_test.shape)

# Correlation Matrix

corr_matrix = housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending = False))

plt.figure(figsize = (12, 8))
sns.heatmap(corr_matrix.abs(), annot = True)

housing_train['TAXRM'] =  housing_train['TAX'] / housing_train['RM']
print(housing_train['TAXRM'])

housing_train.plot(kind = 'scatter', x = 'TAXRM', y = 'MEDV', alpha = 0.6)
print(housing_train.head())

housing['TAXRM'] = housing_train['TAXRM']
print(housing.head())

corr_matrix = housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending = False))

housing_labels = housing_train.pop('MEDV')
print(housing_labels.shape)

housing_y_labels = housing_test.pop('MEDV')
print(housing_y_labels.shape)

print(housing_train.shape)
print(housing_test.shape)


imputer = SimpleImputer(strategy = 'median')
imputer.fit(housing_train)

print(imputer.statistics_)

X = imputer.transform(housing_train)

housing_tr = pd.DataFrame(X, columns = housing_train.columns)
print(housing_tr.describe())


# Creating Pipeline

my_pipeline = Pipeline ([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing_train)
housing_num_ts = my_pipeline.fit_transform(housing_test)

housing_num_tr.shape
housing_num_ts.shape


# Using Model

housing_x_labels = housing.pop('MEDV')
housing_x = my_pipeline.fit_transform(housing)

model = RandomForestRegressor()
model.fit(housing_x, housing_x_labels)

# Training Model using Cross-Validation 

score = cross_val_score(model, housing_x, housing_x_labels, scoring = 'neg_mean_squared_error', cv = 10)
rmse_score = np.sqrt(-score)

print(rmse_score)

def print_scores(score):
    print('Score : ',score)
    print('Mean : ',score.mean())
    print('Standard Deviation : ',score.std())

print_scores(rmse_score)


from joblib import dump 
dump(model, 'HousingPricePred.joblib')