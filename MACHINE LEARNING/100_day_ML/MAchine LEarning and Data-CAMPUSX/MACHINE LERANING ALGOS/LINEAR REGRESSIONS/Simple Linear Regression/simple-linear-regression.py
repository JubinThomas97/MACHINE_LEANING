import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv('placement.csv')
df

# both data are numeracal quant so we can use scatterplt
sns.scatterplot(x=df.cgpa,y=df.package)
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')

X=df.iloc[:,0:1]
y=df.iloc[:,-1]
y
X

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
X_test
y_test

lr.predict(X_test.iloc[0].values.reshape(1,1))
#X_test.iloc[0] is a 1D data ,so we have to reshape it into a 1x1   


plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')


m = lr.coef_   #slope
m
b = lr.intercept_  #intercept
b

y,X
# y = mx + b
m * 6.93 + b     # we giving a new x 
m * 6.22 + b
m * 9.5 + b
