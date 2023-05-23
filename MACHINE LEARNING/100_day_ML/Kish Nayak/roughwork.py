import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston

data=load_boston()
data
df=pd.DataFrame(data.data)
df
df.columns=data.feature_names
df

X = df
y=  data.target 


#  3. train test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
X_train

#4.scaliing

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)
 
# model

from sklearn.linear_model import LinearRegression

regg=LinearRegression()
regg.fit(X_train,y_train)

#cross validation
from sklearn.model_selection import cross_val_score
cross=cross_val_score(regg,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
cross.mean()

#predict
pred=regg.predict(X_test)
pred

# check predict

import seaborn as sns
sns.displot(reg_pred-y_test,kind='kde') 