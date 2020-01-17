import numpy as np
a = [[1,2,3],[3,4,5]]
a = np.asarray(a) #convert normal array to nump array
b = np.array([5,6,7])

# a.shape : (2,3)
# b.shape : (3,)

a = np.reshape(a,(3,2)) # 3 - 2 array converted a : [[1,2],[3,4]...
a = np.ravel(a) #array([1, 2, 3, 3, 4, 5]) 1 dimension it is

b = np.reshape(b,(1,3)) # converted to array([[5, 6, 7]])