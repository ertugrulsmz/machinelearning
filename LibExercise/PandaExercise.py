#%%

import numpy as np
import pandas as pd
seri = pd.Series([1,2,3])
seri
seri.values


#Indexing
x = pd.Series([1,13,5],index=[2,3,5])
x[5]
x = pd.Series([1,13,5], index=['a','b','c'])
x['a']

#Dictionary to Pandas Series
dictionary = {4:1,'b':'x','c':5}
pandadic = pd.Series(dictionary)
pandadic
pandadic['b']



#İndexing
x = pd.Series([1,13,5], index=['b','a',1])
#print(x['a'])
#'a' in x #test index exists
#x['b'] = 195
#x[['a','b']] #selecting multiple element
x[(x>150) & (x<200)] #by condition

#x[1] ve x.loc[] sizin belirttiğiniz indexe göre hareket eder orjinal olana değil.
# x[3:5] is real index movement x.loc[:] will do it as defined. iloc do it for original way.

#DataFrame
seri1 = pd.Series([[1,2,3],[4,5,6]],index=['a','b'])
df = pd.DataFrame(seri1)
'''
              0
a  [1, 2, 3]
b  [4, 5, 6]  df.iloc[0][0][0] = 1
'''
df = pd.DataFrame([[1,2,3],[4,5,6]],columns=['a','b','c'])
# df.iloc[0:1][0:1] => x = df[0:1] then x[0:1]. but , works as it is expected
x = df.loc[df.a>2]
x = df.loc[df['a']>2]
#df['a'] > 2 returns every row index with true or wrong value , then df.loc[true,true,false] way it is selected.
#df.loc[0][df.loc[0]>1 do condition for one row looking columns. 'a' = true , 'b' = false etc returned put in colmns






