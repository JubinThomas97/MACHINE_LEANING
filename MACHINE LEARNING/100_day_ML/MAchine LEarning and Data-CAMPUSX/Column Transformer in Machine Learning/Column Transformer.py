# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:48:00 2023

@author: Justin Thomas
"""

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('covid_toy.csv')
df
df.isnull().sum()
df.value_counts('city')


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['has_covid']),df.iloc[:,-1],test_size=0.2)
X_train
y_train

################################ 1. Aam Zindagi  ########################################

# adding simple imputer to fever col:-
si = SimpleImputer()
X_train_fever = si.fit_transform(X_train[['fever']])

# also the test data
X_test_fever = si.fit_transform(X_test[['fever']])
X_train_fever.shape


# Ordinalencoding -> cough
oe = OrdinalEncoder(categories=[['Mild','Strong']])
X_train_cough = oe.fit_transform(X_train[['cough']])
# also the test data
X_test_cough = oe.fit_transform(X_test[['cough']])
X_train_cough.shape

# OneHotEncoding -> gender,city
ohe = OneHotEncoder(drop='first',sparse=False)
X_train_gender_city = ohe.fit_transform(X_train[['gender','city']])
# also the test data
X_test_gender_city = ohe.fit_transform(X_test[['gender','city']])
X_train_gender_city.shape


# Extracting Age
X_train_age = X_train.drop(columns=['gender','fever','cough','city']).values
# also the test data
X_test_age = X_test.drop(columns=['gender','fever','cough','city']).values
X_train_age.shape
X_train_age


X_train_transformed = np.concatenate((X_train_age,X_train_fever,X_train_gender_city,X_train_cough),axis=1)
# also the test data
X_test_transformed = np.concatenate((X_test_age,X_test_fever,X_test_gender_city,X_test_cough),axis=1)

X_train_transformed.shape



################################ 2. Mentos Zindagi  ########################################
#after train test split
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),                     #1. tranformer namer,object names, column name on whivh function is applied
    ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tnf3',OneHotEncoder(sparse=False,drop='first'),['gender','city'])
],remainder='passthrough')  #passthrough mean do nothing

transformer.fit_transform(X_train).shape
transformer.transform(X_test).shape

