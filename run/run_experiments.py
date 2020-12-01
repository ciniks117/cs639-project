
import sys
sys.path.insert(0, '/home/nikhil/Downloads/pavt/Outside-the-Box')
from run.run_experiment_novelty_variation import *
from run.run_experiment_layer_variation import *
from run.run_experiment_distance import *
from run.run_experiment_other_abstractions import *


def run_all_experiments():
    run_experiment_novelty_variation_all()
    run_experiment_layer_variation_all()
    run_experiment_other_abstractions_all()
    run_experiment_distance()


if __name__ == "__main__":
    run_all_experiments()
