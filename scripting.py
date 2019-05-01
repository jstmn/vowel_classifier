import numpy as np
import time
from scipy.interpolate import interp1d

new_length = 20
# y = np.array([[1, 1, 1], [2,2,2], [3,3,3], [4,4,4], [5,5,5]])
y = [[1, 1, 1], [2,2,2], [3,3,3], [4,4,4], [5,5,5]]
t = np.linspace(0,10, len(y))
t_new = np.linspace(0, 10, new_length )
f = interp1d(t,y, axis=0)
ynew = f(t_new)

print("\nt:\t\t",t)
print("\nt new:\t",t_new)
print("\ny:",y)
print("\ny_new:\n",ynew)
print("\ny_new shape:",ynew.shape)

