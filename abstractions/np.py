# importing numpy as np 
import numpy as np 

arr1 = np.array([]) 
arr1 = np.array([1,0,0,0])[np.newaxis] 
print(arr1.T)
arr2 = np.array([[3, 5, 7, 9]]) 
if arr1.size==0:
    gfg = arr2
else:
    gfg = np.c_[arr1.T, arr2.T]

print (gfg)
print(gfg.shape)
out_arr = np.hstack((arr1.T, arr2.T))
print(out_arr)
