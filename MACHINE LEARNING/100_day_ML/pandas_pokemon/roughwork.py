import pandas 

df=pandas.read_csv('pokemon_data.csv')
df

df.columns
df.loc[df.Name]
df.iloc[:,1]
for i, r in df.iterrows():
    print(i,r)
    
df
df['Total']=df.iloc[:,4:10].sum(axis=1)
df
df.loc[df['Type 1']=="Fire"]

df.sort_values("Name")


cols=list(df.columns)
cols
df=df[cols[0:4] + [cols[-1]] + cols[4:12]]
df
df.columns

df=df.drop(columns=["Total"])
df.columns

