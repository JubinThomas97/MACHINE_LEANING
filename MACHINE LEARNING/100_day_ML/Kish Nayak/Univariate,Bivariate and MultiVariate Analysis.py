import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
df.head()
df.shape        #return (150,5) 150- records and 5 features including an output feature called species

################# UNI VARIENT ANALYSIS #################
#here we use only 1 feature

 # loc[source] Access a group of rows and columns by label(s) or a boolean array.
df_setosa=df.loc[df.species=='setosa']
df_virginica=df.loc[df['species']=='virginica']
df_versicolor=df.loc[df['species']=='versicolor']


#zeros_like() function in Python is used to return an array of zeros ( 0 ) with the same shape 
#and data type as the array passed to it.

plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'o')            # for setosa
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'o')      # for virginica
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'o')    # for versicolor
plt.xlabel("Sepal Length")
plt.show()



####################### Bi VARIENT ANALYSIS #######################
#here we use 2 features

sns.FacetGrid(df,hue='species',size=5).map(plt.scatter,'sepal_length','sepal_width').add_legend()
plt.show()
#A FacetGrid is a multi-axes grid with subplots visualizing the -
#distribution of variables of a dataset and the relationship between multiple variables.

#In seaborn, the hue parameter determines which column in the data frame should be used for colour encoding. 

sns.FacetGrid(df,hue='species',size=5).map(plt.scatter,'petal_length','sepal_width').add_legend()
plt.show()



####################### Multi VARIENT ANALYSIS #######################
sns.pairplot(df,hue='species',size=2)
