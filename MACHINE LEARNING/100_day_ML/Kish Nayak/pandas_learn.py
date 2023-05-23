################################################ PANDAS ################################################

import pandas as pd
import numpy as np

## Playing with Dataframe

df=pd.DataFrame(np.arange(0,20).reshape(5,4),index=['Row1','Row2','Row3','Row4','Row5'],columns=["Column1","Column2","Column3","Coumn4"])
print(df)

## Accessing the elements
df.loc['Row1']


## Check the type
type(df.loc['Row1'])


df.iloc[:,:]
## Take the elements from the Column2
df.iloc[:,1:]

#. loc is primarily used for label indexing and . iloc function is mainly applied for integer indexing.

#convert Dataframes into array
df.iloc[:,1:].values

df['Column1'].value_counts()                    #counts the number of elemets in that column


########  working with csv #########
df=pd.read_csv('mercedesbenz.csv')
print(df)
df.info()               #gives a rough info about it
df.describe()           #more desciption about the data

#Get the unique category counts
df['X0'].value_counts()
df[df['y']>100]                     #returns the values which have y>100, we can also use== df[df.y>100]


#corr() is used to find the pairwise correlation of all columns in the Pandas Dataframe in Python. 
#Any NaN values are automatically excluded. Any non-numeric data type or columns in the Dataframe,it is ignored.
df.corr()


lst_data=[[1,2,3],[3,4,np.nan],[5,6,np.nan],[np.nan,np.nan,np.nan]]         #np.nan given NaN values
df=pd.DataFrame(lst_data)
print(df)

## HAndling Missing Values

##Drop nan values
df.dropna(axis=0)          #drop 0, or 'index' : Drop rows which contain missing values. 1, or 'columns' : Drop columns which contain missing value.
df.dropna(axis=1)


df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'g'],columns=['one', 'two', 'three'])
df2=df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])   #adds extra index values
print(df2)

df2.fillna('Missing')           #adds a specific value given to the NAn values    



############## WORKING WITH CSV FILE ##########
from io import StringIO, BytesIO
data = ('col1,col2,col3\n'
            'x,y,1\n'
            'a,b,2\n'
            'c,d,3')
pd.read_csv(StringIO(data))         #StringIO makes a makshift space in the python and stores that data
## Read from specific columns
df=pd.read_csv(StringIO(data), usecols= ['col1', 'col3'])               #usecol to retrive data 
print(df)

#df.to_csv('Test.csv)               save the csv to that specifi file or create it


## Index columns and training delimiters
 
data = ('index,a,b,c\n'
           '4,apple,bat,5.7\n'
            '8,orange,cow,10')
arr=pd.read_csv(StringIO(data),index_col=0)             #index_col make the given colum as index
print(arr)


data = ('a,b,c\n'
           '4,apple,bat,\n'
            '8,orange,cow,')
pd.read_csv(StringIO(data))
pd.read_csv(StringIO(data),index_col=False)             #creates a new index with "False" keyword


## Combining usecols and index_col
 data = ('a,b,c\n'
           '4,apple,bat,\n'
            '8,orange,cow,')
pd.read_csv(StringIO(data), usecols=['b', 'c'],index_col=False)


## Quoting and Escape Characters¶. Very useful in NLP

data = 'a,b\n"hello, \\"Bob\\", nice to see you",5'
pd.read_csv(StringIO(data),escapechar='\\')


######################################## Read Json to CSV ######################################
'''A JSON object contains data in the form of key/value pair. The keys are strings and the values are the JSON types.
 Keys and values are separated by colon. Each entry (key/value pair) is separated by comma.'''
 
Data = '{"employee_name": "James", "email": "james@gmail.com", "job_profile": [{"title1":"Team Lead", "title2":"Sr. Developer"}]}'
df1=pd.read_json(Data)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)    #if we take header as 0 it will take the first row as header
print(df)

# convert Json to csv     df.to_csv('wine.csv')

# convert Json to different json formats

df1.to_json() 
#'{"employee_name":{"0":"James"},"email":{"0":"james@gmail.com"},"job_profile":{"0":{"title1":"Team Lead","title2":"Sr. Developer"}}}', here 
#we can see '0' comes as key insided the nested dict so to remove it we use orient keyword

df1.to_json(orient='records')


######################################## Reading EXcel Files ######################################

df_excel=pd.read_excel('Excel_Sample.xlsx')


###### Pickling
##### All pandas objects are equipped with to_pickle methods which use Python’s cPickle module to save data structures to disk using the pickle format.

df_excel.to_pickle('df_excel')
df=pd.read_pickle('df_excel')