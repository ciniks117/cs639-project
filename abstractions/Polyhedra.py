from numpy import array, eye, ones, vstack, zeros
from pypoman import plot_polygon, project_polytope

"""
### PART1 
from numpy import array
from pypoman import compute_polytope_vertices

A = array([
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1],
    [1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0],
    [0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0],
    [0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1]])
b = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 1, 2, 3])
vertices = compute_polytope_vertices(A, b)
print(vertices)
"""

### PART2
from numpy import array
from pypoman import compute_polytope_halfspaces
from pypoman import compute_polytope_vertices
import numpy as np
#vertices = map(array, [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1]])
vertices = map(array, [[-1,1], [1,-1], [-1,-1], [1,1]])
A, b = compute_polytope_halfspaces(vertices)
print(A)
print(b)
vert = compute_polytope_vertices(A, b)
print(vert)
point=array([2,0])
print(point)
vert1 = np.vstack((vert,point))
print(vert1)
#import pylab
#pylab.ion()
#pylab.figure()

"""
from matplotlib import pyplot as plt
#fig = plt.figure(figsize =(8,8))
plt.plot(vert)
plt.show()
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
#xy = np.random.random((3, 2))
vertices1 = np.array([[0.02,0.33], [0.04,0.3], [0,0.27],[0,0.3],[0,0.39]])
vertices2 = np.array([[0.3, 0.45], [0.38, 0.51], [0.4, 0.48], [0.52, 0.48]])
xy1=vertices1
xy2=vertices2

num_poly=1
#z = np.random.random(num_poly)
#print(xy)
#print(z)
#self.v = np.array([]) 
#patches = [RegularPolygon((x, y),5, 0.1) for x, y in xy]
patches = [Polygon(np.array(xy1),True),Polygon(np.array(xy2),True)]
collection = PatchCollection(patches,edgecolors ='brown',lw = 2)
fig, ax = plt.subplots()
ax.patch.set(facecolor ='white')
ax.add_collection(collection)
ax.autoscale()
plt.show()


"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
fig, ax = plt.subplots()
patches = []
num_polygons = 2
vertices1 = np.array([[0.7, 0.2], [0.6, 0.2], [0.7, 0.1], [0.8, 0.1],[0.9, 0.2]])
vertices2 = np.array([[0.02,0.33], [0.04,0.3], [0,0.27],[0,0.3],[0,0.39]])
num_sides = 5
#for i in range(2):
#polygon = Polygon(np.random.rand(num_sides ,2), True)
polygon = Polygon(vertices1, True)
patches.append(polygon)
polygon = Polygon(vertices2, True)
patches.append(polygon)
p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
colors = 100*np.random.rand(len(patches))
p.set_array(np.array(colors))
ax.add_collection(p)
plt.show()
"""

#plt.savefig('an2.png')

#plot_polygon(vert)
"""
# PART 3
n = 10  # dimension of the original polytope
p = 2   # dimension of the projected polytope

# Original polytope:
# - inequality constraints: \forall i, |x_i| <= 1
# - equality constraint: sum_i x_i = 0
A = vstack([+eye(n), -eye(n)])
b = ones(2 * n)
C = ones(n).reshape((1, n))
d = array([0])
ineq = (A, b)  # A * x <= b
eq = (C, d)    # C * x == d

# Projection is proj(x) = [x_0 x_1]
E = zeros((p, n))
E[0, 0] = 1.
E[1, 1] = 1.
f = zeros(p)
proj = (E, f)  # proj(x) = E * x + f

vertices = project_polytope(proj, ineq, eq, method='bretl')

if __name__ == "__main__":   # plot projected polytope
    import pylab
    pylab.ion()
    pylab.figure()
    plot_polygon(vertices)
"""
