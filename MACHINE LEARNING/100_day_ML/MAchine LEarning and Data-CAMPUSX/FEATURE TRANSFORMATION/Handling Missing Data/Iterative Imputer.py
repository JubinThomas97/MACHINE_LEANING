# THIS technique is also called MICE- Multivarient Imputation by Chained Equation

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = np.round(pd.read_csv('50_Startups.csv')[['R&D Spend','Administration','Marketing Spend','Profit']]/10000)
np.random.seed(9)
df = df.sample(5)
df

df = df.iloc[:,0:-1]  #isolate profit from rest
df

df.iloc[1,0] = np.NaN
df.iloc[3,1] = np.NaN
df.iloc[-1,-1] = np.NaN
df.head()


# Step 1 - Impute all missing values with mean of respective col

df0 = pd.DataFrame()

df0['R&D Spend'] = df['R&D Spend'].fillna(df['R&D Spend'].mean())
df0['Administration'] = df['Administration'].fillna(df['Administration'].mean())
df0['Marketing Spend'] = df['Marketing Spend'].fillna(df['Marketing Spend'].mean())

# 0th Iteration
df0


# Remove the col1 imputed value
df1 = df0.copy()
df1.iloc[1,0] = np.NaN
df1


#Use first 3 rows to build a model and use the last for prediction
X = df1.iloc[[0,2,3,4],1:3]
X

y = df1.iloc[[0,2,3,4],0]
y


lr = LinearRegression()
lr.fit(X,y)
lr.predict(df1.iloc[1,1:].values.reshape(1,2))


df1.iloc[1,0] = 23.14
df1


# Remove the col2 imputed value
df1.iloc[3,1] = np.NaN
df1

# Use last 3 rows to build a model and use the first for prediction
X = df1.iloc[[0,1,2,4],[0,2]]
X
y = df1.iloc[[0,1,2,4],1]
y

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df1.iloc[3,[0,2]].values.reshape(1,2))

df1.iloc[3,1] = 11.06
df1

# Remove the col3 imputed value
df1.iloc[4,-1] = np.NaN
df1

# Use last 3 rows to build a model and use the first for prediction
X = df1.iloc[0:4,0:2]
X

y = df1.iloc[0:4,-1]
y

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df1.iloc[4,0:2].values.reshape(1,2))

df1.iloc[4,-1] = 31.56
# After 1st Iteration
df1


# Subtract 0th iteration from 1st iteration

df1 - df0

df2 = df1.copy()
df2.iloc[1,0] = np.NaN

df2

X = df2.iloc[[0,2,3,4],1:3]
y = df2.iloc[[0,2,3,4],0]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df2.iloc[1,1:].values.reshape(1,2))

df2.iloc[1,0] = 23.78
df2.iloc[3,1] = np.NaN
X = df2.iloc[[0,1,2,4],[0,2]]
y = df2.iloc[[0,1,2,4],1]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df2.iloc[3,[0,2]].values.reshape(1,2))

df2.iloc[3,1] = 11.22
df2.iloc[4,-1] = np.NaN

X = df2.iloc[0:4,0:2]
y = df2.iloc[0:4,-1]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df2.iloc[4,0:2].values.reshape(1,2))

df2.iloc[4,-1] = 31.56
df2

df2 - df1

df3 = df2.copy()
df3.iloc[1,0] = np.NaN
df3

X = df3.iloc[[0,2,3,4],1:3]
y = df3.iloc[[0,2,3,4],0]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df3.iloc[1,1:].values.reshape(1,2))

df3.iloc[1,0] = 24.57
df3.iloc[3,1] = np.NaN
X = df3.iloc[[0,1,2,4],[0,2]]
y = df3.iloc[[0,1,2,4],1]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df3.iloc[3,[0,2]].values.reshape(1,2))

df3.iloc[3,1] = 11.37
df3.iloc[4,-1] = np.NaN

X = df3.iloc[0:4,0:2]
y = df3.iloc[0:4,-1]

lr = LinearRegression()
lr.fit(X,y)
lr.predict(df3.iloc[4,0:2].values.reshape(1,2))

df3.iloc[4,-1] = 45.53
df2.iloc[3,1] = 11.22
df3

df3 - df2