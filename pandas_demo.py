import numpy as np
import pandas as pd
'''------------------------------Object Creation----------------------------'''
# creating Series by passing a list of values, letting pandas create default integer index
s=pd.Series([1,3,5,np.nan,6,8])
s
# creating Dataframe by passing a numpy array, with a datetime index and labeled columns
dates=pd.date_range('20130101', periods=6)
dates
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD')) #np.random.randn generates standard normal array
df
# creating Dataframe by passing a dict of objects that can be converted to series-like
df2=pd.DataFrame({'A':1.,
                  'B':pd.Timestamp('20130102'),
                  'C':pd.Series(1,index=list(range(4)),dtype=np.float32),
                  'D':np.array([3]*4,dtype='int32'),
                  'E':pd.Categorical(['test','train','test','train']),
                  'F':'foo'})
df2
df2.dtypes # having specific dtypes of each column

'''-----------------------------Viewing Data--------------------------------'''
# see the top & bottom rows of the frame
df.head()
df.tail()
# display the index, columns, and the underlying numpy data
df.index
df.columns
df.values   # array form
# describe function shows a quick statistic summary of data
df.describe()
# transpose
df.T
df.transpose()
# sorting by axis
df.sort_index(axis=1, ascending=False) # sorting by columns' names, returns a new dataframe
# sorting by values
df.sort_values(by='B')

'''----------------------------Selection------------------------------------'''
# selecting a single column, which yields a Series, equivalent to df.A
df['A']
df.A
# selecting via [], which slices the rows
df[0:3]
df['20130102':'20130104'] # note endpoints are included since its label

# selection by label, 'labels' means the names of indices and columns
df.loc[dates[0]] # getting a cross section using a label
df.loc[:,['A','B']] # selecting on a multi-axis by label
df.loc['20130102':'20130104',['A','B']] # showing label slicing, both endpoints included
df.loc['20130102',['A','B']] # reduction in dimension, returns a 1-D Series
df.loc[dates[1],'A'] # to get a scalar value
df.at[dates[1],'A'] # same thing

# selection by position
df.iloc[3]
df.iloc[3:5,0:2] # integer slicing, similar to numpy
df.iloc[[1,2,4],[0,2]] # list of integer positions, similar to numpy
df.iloc[1:3,:] # slicing rows explicitly
df.iloc[:,1:3] # slicing columns explicitly
df.iloc[1,1] # getting a value explicitly
df.iat[1,1] # same thing

# boolean indexing 
df[df.A>0] # using a single column's value to select
df[df>0]  # selecting values from a DataFrame where a boolean condition is met

# using the isin() method for filtering
df2=df.copy()
df2['E']=['one','one','two','three','four','three']
df2
df2[df2['E'].isin(['two','four'])]

'''--------------------------Setting--------------------------------------'''
s1=pd.Series([1,2,3,4,5,6,7],index=pd.date_range('20130102',periods=7))
s1
df['F']=s1 # setting a new column automatically aligns the data by the indexes
df.at['20130101','F']=0 # setting value by label
df.iat[0,1]=0 # setting value by position
df.loc[:,'D']=np.array([5]*len(df)) # setting value by assigning with an array
# a where operation with setting
df2=df.copy()
df2[df2>0]=-df2# note np.nan by default not included in computations
df2

'''---------------------------Missing Data---------------------------------'''
# pandas primarily uses np.nan to represent missing data, it's by default not included in computations
# reindexing allows change/add/delete the index on a specified axis,returns a copy of the data
df1=df.reindex(index=dates[0:4],columns=list(df.columns)+['E'])
df1.loc[dates[0:2],'E']=1 # equivalent to df1.loc[dates[0]:dates[1],'E']=1
df1
# to drop the rows that have missing data
df1.dropna(how='any') # returns a new DataFrame
# filling missing data
df1.fillna(value=888) # returns a new DataFrame
# to get the boolean mask where value are nan
pd.isnull(df1)

'''------------------------------Operations--------------------------------'''
#--------- stats
df.mean()
df.mean(1) # along columns
# operating with objects that have different dimensionality
# automatically broadcasts along the specified dimension
s=pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
s
df.sub(s,axis='index') #axis to match Series index on
#--------- Apply
df.apply(np.cumsum,axis=0) # axis by default is 0
df.apply(lambda x: x.max()-x.min())
# Histogramming
s=pd.Series(np.random.randint(0,7,size=10)) # right endpoint is not included
s
s.value_counts()
#------------string methods
s=pd.Series(['A','B','C','Aaba','Baca',np.nan,'CABA','dog','cat'])
s.str.lower()
#---------- Merge
# 1.concat
df=pd.DataFrame(np.random.randn(10,4))
df
pieces=[df[0:3],df[3:6],df[6:10]]
pieces
pd.concat(pieces)  # axis by default is 0,note argument is a list
# 2.join: SQL style merges
#example1:
left=pd.DataFrame({'key':['foo','foo'],'lval':[1,2]})
right=pd.DataFrame({'key':['foo','foo'],'rval':[4,5]})
left
right
pd.merge(left,right,on='key')
#example2:
left=pd.DataFrame({'key':['foo','bar'],'lval':[1,2]})
right=pd.DataFrame({'key':['foo','bar'],'rval':[4,5]})
left
right
pd.merge(left,right,on='key')
#-------- Append
df=pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D'])
df
s=df.iloc[3]
df.append(s,ignore_index=True) # returns a new dataframe
#------ Grouping
# 'group by':splitting data,applying a function to each group,combining the results
df=pd.DataFrame({'A':['foo','bar','foo','bar','foo','bar','foo','bar'],
                 'B':['one','one','two','three','two','two','one','three'],
                 'C':np.random.randn(8),
                 'D':np.random.randn(8)})
df
df.groupby('A').sum() # group and applying function sum to the resulting groups
df.groupby(['A','B']).sum() # group by multiple columns forms a hierarchical index,which we apply the function
#------ Reshaping
# stack
tuples=list(zip(['bar','bar','baz','baz',
                 'foo','foo','qux','qux'],
                ['one','two','one','two',
                 'one','two','one','two']))
index=pd.MultiIndex.from_tuples(tuples,names=['first','second'])
df=pd.DataFrame(np.random.randn(8,2),index=index,columns=['A','B'])
df
# the stack method 'compresses' a level in the DataFrame's columns
stacked=df.stack()
stacked
stacked.unstack() # the inverse operation of stack(),unstack the last level by default
stacked.unstack(0) # unstack the first level
# pivot tables
df=pd.DataFrame({'A':['one','one','two','three']*3,
                 'B':['A','B','C']*4,
                 'C':['foo','foo','foo','bar','bar','bar']*2,
                 'D':np.random.randn(12),
                 'E':np.random.randn(12)})
df
pd.pivot_table(df,values='D',index=['A','B'],columns=['C'])

'''--------------------------Time Series-----------------------------------'''
# perform resampling operations during frequency conversion
rng=pd.date_range('1/3/2012',periods=100,freq='S')
rng
ts=pd.Series(np.random.randint(0,500,len(rng)),index=rng)
ts.resample('1Min').sum()
# time zone representation
rng=pd.date_range('3/6/2012',periods=5,freq='D')
ts=pd.Series(np.random.randn(len(rng)),index=rng)
ts
ts_utc=ts.tz_localize('UTC')
ts_utc
ts_utc.tz_convert('US/Eastern') # convert to another time zone
# converting between time span representation
rng=pd.date_range('1/1/2012',periods=5,freq='M')
ts=pd.Series(np.random.randn(len(rng)),index=rng)
ts
ps=ts.to_period() # period means a period of time
ps
ps.to_timestamp() # timestamp means a time ponit
prng=pd.period_range('1990Q1','2000Q4',freq='Q-NOV')
prng
ts=pd.Series(np.random.randn(len(prng)),index=prng)
ts.index=(prng.asfreq('M','e')+1).asfreq('H','s')+9 
ts

'''------------------------------Categorical--------------------------------'''
df=pd.DataFrame({'id':[1,2,3,4,5,6],'raw_grade':['a','b','b','a','a','e']})
# convert the raw grade to a catagoracal data type
df['grade']=df['raw_grade'].astype('category')
df['grade']
# rename the categories to meaningful names(the assignment is inplace)
df['grade'].cat.categories=['very good','good','very bad']
df
# reorder the categories and simultaneously add the missing categories
df['grade']=df['grade'].cat.set_categories(['very bad','bad','medium','good','very good'])
# sorting is by order in the categories:
df.sort_values(by='grade')
# grouping by a categorical column
df.groupby('grade').size()

'''-------------------------------Plotting----------------------------------'''
ts=pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2011',periods=1000))
ts=ts.cumsum()
ts.plot()
# On DataFrame, plot() is convenience to plot all of the columns with labels
df=pd.DataFrame(np.random.randn(1000,4),index=ts.index,columns=['A','B','C','D'])
df=df.cumsum()
df.plot()

'''-----------------------------Getting Data In/Out------------------------'''
#-----csv
# writing to a csv file
df.to_csv('foo.csv')
# reading from csv file
pd.read_csv('foo.csv',index_col= 0)
#-----excel
# wrting to an excel file
df.to_excel('foo.xlsx',sheet_name='Sheet1')
# reading from an excel file
pd.read_excel('foo.xlsx','Sheet1',index_column=None,na_values=['NA'])
