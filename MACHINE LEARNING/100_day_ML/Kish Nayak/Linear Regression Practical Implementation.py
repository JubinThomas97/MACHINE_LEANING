import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


################# 1. DATAFRAME

from sklearn.datasets import load_boston
df=load_boston()
df

#convert the array into dataframe
dataset=pd.DataFrame(df.data)
dataset

dataset.info()

# Adding the missing column names

dataset.columns=df.feature_names
dataset.head()


################# 2. Independent features and dependent features
X=dataset
y=df.target
X


################# 3. train test split 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)      #mostly random_state is kept as 42
X_train

################# 4.standardizing the dataset
""" Standardization is an important technique that is mostly performed as a pre-processing step before many machine learning models, 
to standardize the range of features of an input data set.And to improve the output

BUT-
Logistic regressions and tree-based algorithms such as decision trees, random forests and gradient boosting are not 
sensitive to the magnitude of variables. So standardization is not needed before fitting these kinds of models.
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


################# 5.Applying Linear Regression Model
from sklearn.linear_model import LinearRegression

regression=LinearRegression()
regression.fit(X_train,y_train)



################# 6.Cross Validation - A type of hyper-parameter tuning
'''Cross-validation is a technique for evaluating ML models by training several ML models on subsets of the available
 input data and evaluating them on the complementary subset of the data. Use cross-validation to detect overfitting, ie,
 failing to generalize a pattern.
'''
from sklearn.model_selection import cross_val_score

mse=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
np.mean(mse)                # as mean is really small no. so its really good -25.550660791660782


################# 7.Prediction 
reg_pred=regression.predict(X_test)
reg_pred


######## 8. Checking

import seaborn as sns
sns.displot(reg_pred-y_test,kind='kde')   # we are doing the differnce btw predicted and actual 
#as the grap is mostly between -10 to 10 with some variance to the left therefore the output is really good


from sklearn.metrics import r2_score
score=r2_score(reg_pred,y_test)
score                                #0.6693702691495595
