# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:10:47 2023

@author: Justin Thomas
"""

############################## LET'S STUDY PANDAS THROUGH POKEMON ##############################

import pandas as pd

df=pd.read_csv('pokemon_data.csv')
df.head(3)                              #prints the first 3 rows
df.tail(3)                              #prints the last 3 rows

# to read txt files pd.read_csv("--.txt",delimiter='\t')


########### 1. READING DATA IN PANDAS ###########

# Read Headers
df.columns

# Read each Column
df.Name         #but this doest work if there are any space then df['Nmaes if']

df['Name']

df.Name[0:5]

print(df[['Name', 'Type 1', 'HP']])



# Read Row
df.iloc[1]    # iloc- integer location
df.iloc[2,1] 


# itterating through each row
for index, row in df.iterrows():
    print(index,row)


for index, row in df.iterrows():
    print(index,row['Name'])


# to find textual info in rows
df.loc[df['Type 1']=="Fire"]


########### 2. SORTING/DESCRIBING DATA  ###########

df.describe()

df.sort_values('Name',ascending=False)   #sorts the data decending namewise



########### 3. MAKING CHANGES TO THE DATA  ###########
df.columns

df['Total']=df.iloc[:,4:10].sum(axis=1)  # axis 1 = add horizontal
df

#To drop a column ---- df=df.drop(columns=['Total'])

# to change the "total" column to left side
cols=list(df.columns)
df=df[cols[0:4] + [cols[-1]] + cols[4:12]]
df
df.columns


######## 4. Saving a files ########

# df.to_csv('modified.csv', index=False)

#df.to_excel('modified.xlsx', index=False)


######## 5. Filtering Data ########
new_df=df.loc[(df['Type 1']=='Grass') & (df['Type 2']=='Poison') & (df['HP'] > 70)]            #and & , or |
new_df

'''
       #                   Name Type 1  ... Speed  Generation  Legendary
2      3               Venusaur  Grass  ...    80           1      False
3      3  VenusaurMega Venusaur  Grass  ...    80           1      False
50    45              Vileplume  Grass  ...    50           1      False
77    71             Victreebel  Grass  ...    70           1      False
652  591              Amoonguss  Grass  ...    30           5      False
'''
#see here the index is so random, so make new index

new_df.reset_index(drop=True, inplace=True)    #this drops the (#)index
new_df



### Filter out mega named pokemon

df.loc[df['Name'].str.contains('Mega')]

# to drop all the words with mega

df.loc[~df['Name'].str.contains('Mega')]






######## 6. Conditional change of Data ########



df.loc[df['Type 1'] == "Fire",'Type 1']="Flamer"
df

# to change back----- df.loc[df['Type 1'] == "Flamer",'Type 1']="Fire"
# to make all the  fire pokemon as legendary pokemonas --  df.loc[df['Type 1'] == "Fire",'Legenday']="True"




######################### GROUP BY DATA #########################

df.groupby(['Type 1']).mean()

df.groupby(['Type 1']).mean().sort_values('Defense',ascending=False) #sort decensending order wrt to defense with mean of type1


df.groupby(['Type 1']).sum()        # sums the data wrt to type 1
df.groupby(['Type 1']).count()      # count the data wrt to type 1

df['count']             # make a colum called count and assigend it all to 1
df.groupby(['Type 1']).count()['count'] # used the above count colum to count the no.of each instances

df.groupby(['Type 1','Type 2']).count()['count']




