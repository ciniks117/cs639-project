from copy import deepcopy
from matplotlib.patches import Rectangle
import random
from .PointCollection import PointCollection
from utils import *
import numpy as np
import scipy.linalg
from numpy import array
from pypoman import compute_polytope_halfspaces
from pypoman import compute_polytope_vertices
from pypoman import project_polytope
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class Star(PointCollection):
    def __init__(self, dimension):
        super().__init__()
        print(" <<<<<<< IN STAR >>>>>>>>>>>>>")
        self.v = []
        self.c = []
        self.d = []

    def __str__(self):
        if self.isempty():
            return "raw Star"
        return "  Star(m={:d})".format(len(self.points))

    def create(self, points):
        #super().create(deepcopy(points[0]))
        #vertices = map(array, [[-1,1], [1,-1], [-1,-1], [1,1]])
        vertices = map(array, [[points[0],points[1]], [points[0]*1.01,points[1]], [points[0],0.99*points[1]], [points[0],1.01*points[1]]])
        #vertices = map(array,points)
        print("Points >>> ")
        print(points)
        A, b = compute_polytope_halfspaces(vertices)
        print(A)
        print(b)
        self.v = np.array([[0,0],[1,0],[0,1]]) 
        self.c = A
        self.d = b
        #print(self.c)
        #print(self.d)

    def add(self, point):
        #super().add(point)
        print(self.v)
        print(self.c)
        print(self.d)
        vert = compute_polytope_vertices(self.c,self.d)
        #plot_polygon(vert)
        print(" adding point >>>")
        print(point)
        #print(vert)
        new_vert = np.vstack((vert,point))
        new_A, new_b = compute_polytope_halfspaces(new_vert)
        self.c = new_A
        self.d = new_b

    
    
    """
    def create(self, *args):
        super().create(*args)
        if len(args) == 5:
            v = args[0]
            c = args[1]
            d = args[2]
            pred_lb = args[3]
            pred_ub = args[4]

            [n_rows_v, n_cols_v] = np.shape(v)
            [n_rows_c, n_cols_c] = np.shape(c)
            [n_rows_d, n_cols_d] = np.shape(d)
            [n_rows_plb, n_cols_plb] = np.shape(pred_lb)
            [n_rows_pub, n_cols_pub] = np.shape(pred_ub)

            if n_cols_v != n_cols_c + 1:
                raise Exception("Inconsistency between basic matrix and constraint matrix")

            if n_rows_c != n_rows_d:
                raise Exception("Inconsistency between constraint matrix and constraint vector")

            if n_cols_d != 1:
                raise Exception("constraint vector should have one column")

            if (n_cols_plb != 1) or (n_cols_pub != 1):
                raise Exception("predicate lower- or upper-bounds vector should have exactly one column")

            if (n_rows_plb != n_rows_pub) or (n_rows_plb != n_cols_c):
                raise Exception("Inconsistency between number of predicate variables and predicate lower- or upper-bounds vector")
            
            self.v = star.v 
            self.c = star.c
            self.d = star.d
            self.dimension = star.dimension
            self.n_var = star.n_var
            self.predicate_lower_bound = -np.ones((self.n_var, 1))
            self.predicate_upper_bound = np.ones((self.n_var, 1))

        elif len(args) == 3:
            v = args[0]
            c = args[1]
            d = args[2]

            [n_rows_v, n_cols_v] = np.shape(v)
            [n_rows_c, n_cols_c] = np.shape(c)
            [n_rows_d, n_cols_d] = np.shape(d)

            if n_cols_v != n_cols_c + 1:
                raise Exception("Inconsistency between basic matrix and constraint matrix")

            if n_rows_c != n_rows_d:
                raise Exception("Inconsistency between constraint matrix and constraint vector")

            if n_cols_d != 1:
                raise Exception("Constraint vector should have only one column")
            
            self.v = v
            self.c = c
            self.d = d
            self.dimension = n_rows_v
            self.n_var = n_cols_c
            
        elif len(args) == 2:
            #construct a star from a lower bound and an upper bound (basically using a box)
            lower_bound = args[0]
            upper_bound = args[1]

            box = Box(lower_bound, upper_bound)
            star = box.toStar()
            self.v = star.v 
            self.c = star.c
            self.d = star.d
            self.dimension = star.dimension
            self.n_var = star.n_var
            self.predicate_lower_bound = -np.ones((self.n_var, 1))
            self.predicate_upper_bound = np.ones((self.n_var, 1))

        elif len(args) == 0:
            #create an empty star
            self.v = np.array([]) 
            self.c = np.array([]) 
            self.d = np.array([]) 
            self.dimension = 0
            self.n_var = 0
        else:
            print(len(args))
            print(args)
            raise Exception("Incorrect number of arguments. Please retry.")


    """

    def scalarMap(self, alp_max):
            """
            % @a_max: maximum value of a
            % @S: new Star
            
            % note: we always require that alp >= 0
            % =============================================================
            % S: x = alp*c + V* alph * a, Ca <= d
            % note that:   Ca <= d -> C*alph*a <= alp*a <= alp_max * d
            % let: beta = alp * a, we have
            % S := x = alp * c + V * beta, C * beta <= alp_max * d,
            %                              0 <= alp <= alp_max
            % Let g = [beta; alp]
            % S = Star(new_V, new_C, new_d), where:
            %   new_V = [0 c V], new_C = [0 -1; 0 1; 0 C], new_d = [0; alpha_max; alp_max * d]
            %       
            % S has one more basic vector compared with obj
            % =============================================================
            """
            dim=2
            new_c = np.zeros([dim,1])
            new_V = np.c_[self.v, new_c]
            tmp = np.array([-1, 1])[np.newaxis]
            new_C = scipy.linalg.block_diag(self.c, tmp.T)
            new_d = np.vstack((alp_max*self.d, 0, alp_max))
            S = Star(new_V, new_C, new_d)
            return S

        
# concatenate many stars
def concatenateStars(stars):
            # @stars: an array of stars
            
            new_c = np.array([])
            new_V = np.array([])
            new_C = np.array([])
            new_d = np.array([])
            nVar=2 
            n = len(stars);
            
            for i in range(0,n):
                #if ~isa(stars(i), 'Star'):
                #    error('The %d th input is not a Star', i)
                #print(stars[0].v)
                tmp = np.array(stars[i].v[:,0])[np.newaxis]
                if len(new_c)==0:
                    new_c=tmp.T
                else:
                    new_c = np.vstack((new_c, tmp.T))
                tmp=stars[i].v[:,1:nVar+1]
                if new_V.size==0:
                    new_V=tmp
                else:
                    new_V = scipy.linalg.block_diag(new_V, tmp)
                tmp=stars[i].c
                if new_C.size==0:
                    new_C=tmp
                else:
                    new_C = scipy.linalg.block_diag(new_C,tmp)
                tmp = np.array(stars[i].d)[np.newaxis]
                if len(new_d)==0:
                    new_d=stars[i].d
                else:
                    #print(stars[i].d)
                    #print(new_d)
                    new_d = np.vstack((new_d, stars[i].d))
               
            print(new_c)
            print(new_V)
            tmp=np.c_[new_c,new_V] 
            print(tmp)
            print(stars[0].v.shape)
            print(stars[1].v.shape)
            print(stars[0].c.shape)
            print(stars[1].c.shape)
            print(stars[0].d.shape)
            print(stars[1].d.shape)
            print(tmp.shape)
            print(new_C.shape)
            print(new_d.shape)
            S = Star(tmp, new_C, new_d)
            return S
           
       
#def plot(self, dims, color, epsilon, epsilon_relative, ax):
def plot(stars):
    vert=[]
    for star in stars:
        b = star.v[0,:]
        W = star.v[1:len(star.v),:]
        print(star.v)
        print(W)
        print(b)
        #tmp = np.array(stars[i].v[:,0])[np.newaxis]
        ineq = (star.c, star.d) 
        proj = (W,b)  # proj(x) = E * x + f
        vertices = project_polytope(proj, ineq)
        #fig = plt.figure(figsize =(8,8))
        print(" our star is : >> ")
        print(vertices)
        vert.append(vertices)
        #plt.plot(vertices)
        #plt.show()
        #plt.savefig('an2.png')
    patches=[]
    for i in range(len(vert)):
        patches += [Polygon(np.array(vert[i]),True)]
    collection = PatchCollection(patches,edgecolors ='brown',lw = 2)
    fig, ax = plt.subplots()
    ax.patch.set(facecolor ='white')
    ax.add_collection(collection)
    ax.autoscale()
    plt.show()

"""
        xy=vertices

        num_poly=1
        z = np.random.random(num_poly)
        print(xy)
        print(z)
        #self.v = np.array([]) 
        #patches = [RegularPolygon((x, y),5, 0.1) for x, y in xy]
        patches = [Polygon(np.array(xy),True)]
        #collection = PatchCollection(patches,edgecolors ='brown',lw = 2)
        collection = PatchCollection(patches,edgecolors ='black')
        fig, ax = plt.subplots()
        ax.patch.set(facecolor ='white')
        ax.add_collection(collection)
        ax.autoscale()
        plt.show()
    
"""
