import pandas as pd
import numpy as np

M = np.array([[10,10,10],[7,12,12],[-4,2,16]])
df = pd.DataFrame(M, columns = ["Low","Moderate","High"], 
                  index = ["Small Facility","Medium Facility","Large Facility"])


M2 = np.array([[40,45,5],[70,30,-13],[53,45,-5]])
df2 = pd.DataFrame(M2, columns = ["Low","Moderate","High"], 
                  index = ["Small Facility","Medium Facility","Large Facility"])






def minimaxCalculator(df):
    m = df.values
    maxColumn = np.amax(m, axis=0)
    dif = maxColumn - m
    maxregret = np.amax(dif,axis = 1)
    x = np.amin(maxregret,axis = 0)
    index = np.where(maxregret == x )
    indexvalue = index[0]
    indexname = df.index.values[indexvalue]
    return (indexname,x)

