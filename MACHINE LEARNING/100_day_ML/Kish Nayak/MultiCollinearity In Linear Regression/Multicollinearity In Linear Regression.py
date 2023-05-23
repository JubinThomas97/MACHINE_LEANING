############################## Multicollinearity In Linear Regression ##############################


import pandas as pd
import statsmodels.api as sm

df_adv = pd.read_csv('Advertising.csv', index_col=0)
df_adv

X = df_adv[['TV', 'radio','newspaper']]
y = df_adv['sales']
df_adv.head()

# creating a beta0 column for ols -ordinary least sqrae
X = sm.add_constant(X)
X

## fit a OLS model with intercept on TV and Radio
model= sm.OLS(y, X).fit()
model.summary()
# here {newspaper  has P value  0.860}, R- square error is close to 1 so the model fits better

import matplotlib.pyplot as plt
X.iloc[:,1:].corr()



#############

df_salary = pd.read_csv('Salary_Data.csv')
df_salary.head()

X = df_salary[['YearsExperience', 'Age']]
y = df_salary['Salary']

## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
model= sm.OLS(y, X).fit()
model.summary()
# here Age  has a p value of  0.165 , which is higher than the p value of YearsExperience so it can be dropped


X.iloc[:,1:].corr()
