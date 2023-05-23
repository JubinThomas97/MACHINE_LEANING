# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 11:57:57 2023

@author: Justin Thomas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('titanic_train.csv')
df
df.columns


df.isnull()
df.isnull().sum()

sns.countplot(x=df.Survived)
sns.countplot(df.Survived,hue=df.Sex)
sns.countplot(df.Survived,hue=df.Pclass)

df.corr()
sns.heatmap(df.corr())
sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=40)

sns.histplot(df.Age.dropna(),bins=20)

df[['Age','Pclass']]
pd.isnull(df.Age)

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24
    else:
        return Age

df.Age= df[['Age','Pclass']].apply(impute_age,axis=1)
df.Age

