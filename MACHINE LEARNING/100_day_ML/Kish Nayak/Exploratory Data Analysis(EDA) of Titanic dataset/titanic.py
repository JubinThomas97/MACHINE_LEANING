###################################  LOGISTIC REGRESSION WITH PYTHON ###################################

'''
For this lecture we will be working with the Titanic Data Set from Kaggle. This is a very famous data set and very often is a 
student's first step in machine learning! We'll be trying to predict a classification- survival or deceased. 
Let's begin our understanding of implementing Logistic Regression in Python for classification. We'll use a "semi-cleaned" version of the 
titanic data set, if you use the data set hosted directly on Kaggle, you may need to do some additional cleaning not shown in this lecture notebook.

'''

#Let's import some libraries to get started!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Let's start by reading in the titanic_train.csv file into a pandas dataframe.
train=pd.read_csv('titanic_train.csv')
print(train)


##Exploratory Data Analysis
'''
Let's begin some exploratory data analysis! We'll start by checking out missing data!
Missing Data-
We can use seaborn to create a simple heatmap to see where we are missing data! '''

train.isnull()

sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis') #cmap is the colour scheme and viridis is the colour map
#the heatmap gives us which of columns have the max NaN values. here we can se cabin and age has the max NaN values.. cabin>>>


'''
Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable 
replacement with some form of imputation. Looking at the Cabin column,it looks like we are just missing too much of that
 data to do something useful with at a basic level. 
 1-survived , 0-passed away
We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
Let's continue on by visualizing some more of the data! Check out the video for full explanations over these plots,
 this code is just to serve as reference.
'''
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)

# Now differentiating between sexes
#In seaborn, the hue parameter determines which column in the data frame should be used for colour encoding.
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

# Now differentiating between Passenger class 1st,2nd and 3rd
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)

sns.countplot(x='SibSp',data=train)         #wrt sibling and spouse


sns.distplot(train['Fare'].dropna(),kde=False,color='darkred',bins=10)


########################## DATA CLEANING ##########################

# first we will remove the null values .We want to fill in missing age data instead of just dropping the missing age data rows. 
#One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class.
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

#here in box plot the values starts from 20% the up 50% and then up 75% fo the 3 division of the box (from down to up)

#We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#here .apply will take in values of age and pclass and apply the values to the function and save it to the train['age']


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.
train.drop('Cabin',axis=1,inplace=True)
#so again
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.dropna(inplace=True)
    
#####   Converting Categorical Features
#We'll need to convert categorical features to dummy variables using pandas! (here sex is converted into 1 and 0)
#Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.
# also called as 1hot encoding
train.info()
pd.get_dummies(train['Embarked'],drop_first=True)
# here we can see we droped the first column p because it can be represented by Q(1,0) ans S(0,1). so no need of P


#simillarly for sex also
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

#droping the column that are not requred
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train
train = pd.concat([train,sex,embark],axis=1)
train.head()



#################  Building a Logistic Regression model  #################
##Let's start by splitting our data into a training set and test set 
#(there is another test.csv file that you can play around with in case you want to use all this data for training).

#  Train Test Split

train.drop('Survived',axis=1).head()
train['Survived'].head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)



























