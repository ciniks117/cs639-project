
import sys                                                                         
sys.path.insert(0, '/home/nikhil/Downloads/pavt/Outside-the-Box') 
from utils import *
from abstractions import *
from trainers import *
from monitoring import *
from run.Runner import run

from abstractions.ConvexHull import ConvexHull


def run_script():
    chull = ConvexHull(2)
    chull.create([[0., 0.1], [0., 1.], [1., 1.], [1., 0.], [0.5, 0.5]])
    cfun = euclidean_distance
    print(chull.contains([0., 0.],cfun))
    print(chull.contains([0., 1.],cfun))
    print(chull.contains([1., 1.],cfun))
    print(chull.contains([1., 0.],cfun))
    print(chull.contains([0.5, 0.5],cfun))
    print(chull.contains([1.5, 1.5],cfun))
    print(chull.contains([-0.5, -0.5],cfun))
    chull.plot([0, 1], "r", 0, False, ax=plt.figure().add_subplot())
    plt.draw()
    plt.pause(0.0001)

if __name__ == "__main__":
    run_script()
