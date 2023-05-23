############################################ SEABORN TUTORIAL ############################################

'''
Distribution plots

*)distplot
*)joinplot        USED IN BIVARIANT ANALYSIS  RELATIONSHIP B/W 2 VARIABLES
*)pairplot        USED IN  ANALYSIS  RELATIONSHIP B/W 4.... VARIABLES

Practise problem on IRIS Dataset
'''

import seaborn as sns

df=sns.load_dataset('tips')

print(df)

'''
Correlation with Heatmap
A correlation heatmap uses colored cells, typically in a monochromatic scale, 
to show a 2D correlation matrix (table) between two discrete dimensions or event types. It is very important in Feature Selection
'''
df.corr()

'''
output-
            total_bill       tip      size
total_bill    1.000000  0.675734  0.598315
tip           0.675734  1.000000  0.489299
size          0.598315  0.489299  1.000000

it means for total bill , if total bill is increasing your tip will also increase and so on

'''

sns.heatmap(df.corr())


##### JointPlot
'''A join plot allows to study the relationship between 2 numeric variables. 
The central chart display their correlation. It is usually a scatterplot, a hexbin plot, 
a 2D histogram or a 2D density plot'''
sns.jointplot(x='tip',y='total_bill',data=df,kind='hex')
sns.jointplot(x='tip',y='total_bill',data=df,kind='reg')


######## Pair plot
'''A “pairs plot” is also known as a scatterplot, in which one variable in the same data row is matched with another variable's value,
 like this: Pairs plots are just elaborations on this, showing all variables paired with all the other variables'''
 
sns.pairplot(df)
 
df['sex'].value_counts()
sns.pairplot(df,hue='sex')
 
 
 
################### Categorical Plots ###################

'''
Seaborn also helps us in doing the analysis on Categorical Data points.
 In this section we will discuss about

> boxplot
> violinplot
> countplot
> bar plot

'''

## Count plot   used against count vs the specified object

sns.countplot(x='sex',data=df)
sns.countplot(y='sex',data=df)
sns.countplot(x='day',data=df)


## Bar plot
sns.barplot(x='total_bill',y='sex',data=df)





## Practise Homework

iris = sns.load_dataset('iris')
print(iris)
iris.corr()
sns.heatmap(iris.corr())
sns.jointplot(x='sepal_length',y='sepal_width',data=iris,kind='reg')

sns.pairplot(iris)
iris['species'].value_counts()
sns.pairplot(iris,hue='species')
sns.countplot(x='species',data=iris)