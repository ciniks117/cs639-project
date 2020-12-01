"""
import matplotlib.pyplot as plt
coord = [[1,1], [2,1], [2,2], [1,2], [0.5,1.5]]
coord.append(coord[0]) #repeat the first point to create a 'closed loop'
xs, ys = zip(*coord) #create lists of x and y values
plt.figure()
plt.plot(xs,ys) 
plt.show() # if you need...
"""

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import numpy as np
xy = np.random.random((3, 2))
z = np.random.random(3)
patches = [RegularPolygon((x, y),5, 0.1) for x, y in xy]
collection = PatchCollection(patches,array = z,edgecolors ='brown',lw = 2)
fig, ax = plt.subplots()
ax.patch.set(facecolor ='white')
ax.add_collection(collection)
ax.autoscale()
plt.show()

