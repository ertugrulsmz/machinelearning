import pandas as pd
import numpy as np

df = pd.read_csv('MA.csv')
x_value_array = df['x'].values


def ma(x,k):
    x_pred = np.full(x.shape, np.nan)
    for t in range(k,x.size):
        x_pred[t] = np.mean(x[(t-k):t])
    return x_pred

def mape(y,ypred):
    return np.mean(np.abs(y-ypred)/np.abs(y))*100


def optimum_k_value(x_value_array): 
    min_error = 0
    kvalue = 0
    for k in range(2,16):
        pred = ma(x_value_array,k)
        pred = pred[k:] #get rid of from null values
        error_mean = mape(x_value_array[k:],pred)
        if k == 2:
            min_error = error_mean
            kvalue = k
        else:
            if error_mean < min_error:
                min_error = error_mean
                kvalue = k
    return (min_error,kvalue)                
    
    
#function call example
bestvalues = optimum_k_value(x_value_array)
print(" Lowest error is :",bestvalues[0], " K value : ",bestvalues[1])    

        

