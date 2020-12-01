import sys                                                                         
sys.path.insert(0, '/home/nikhil/Downloads/pavt/Outside-the-Box') 
from utils import *
from abstractions import *
from trainers import *
from monitoring import *
from run.Runner import run

from abstractions.ConvexHull import ConvexHull
from tensorflow_core.python.keras.models import load_model
from models import *

def run_script():
    # options
    seed = 0
    data_name = "MNIST"
    classes = [0, 1]
    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    data_train_monitor = DataSpec(n=300, randomize=True, classes=[0, 1])
    data_test_monitor = DataSpec(n=100, randomize=False, classes=[2])
    data_run = DataSpec(n=200, randomize=True, classes=[0, 1, 9])
    model_name = "MNIST"
    model_path = "MNIST_2-model.h5"
    n_epochs = 2
    batch_size = 128
    score_fun = F1Score()
    # confidence_thresholds = [0]
    confidence_thresholds = uniform_bins(1, max=1)
    alpha = 0.95
    alphas = [0.99, 0.95, 0.9, 0.5]

    #m = MNIST_CNY19(classes) 
    # model trainer
    model_trainer = StandardTrainer()

    # abstractions
    confidence_fun_euclidean = euclidean_distance
    #confidence_fun_half_space = halfspace_distance

    box_abstraction = BoxAbstraction(confidence_fun_euclidean, epsilon=0.0)
    #box_abstraction_hs = BoxAbstraction(confidence_fun_half_space, epsilon=0.0)

    layer2dimensions = {-3: [68, 69], -2: [12, 14]}
    layer2abstraction = {-2: box_abstraction}
    print(layer2abstraction)
    print(layer2abstraction[-2].sets)
    print(layer2abstraction[-2].dim)
    print(layer2abstraction[-2].epsilon)
    print(layer2abstraction[-2].epsilon_relative)
    print(layer2abstraction[-2].confidence_fun)
    monitor1 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    monitor1_w_novelties = Monitor(layer2abstraction, score_fun, layer2dimensions, is_novelty_training_active=True)

    monitors = [monitor1, monitor1_w_novelties]
    monitor_manager = MonitorManager(monitors, clustering_threshold=0.1, n_clusters=3)

    print("***************************************************")
    for m in monitor_manager._monitors:
        print(m._layer2abstraction[-2].dim)
    print(monitor_manager._layers)
    history_run, histories_alpha_thresholding, novelty_wrapper_run, novelty_wrappers_alpha_thresholding, \
        statistics = evaluate_all(seed=seed, data_name=data_name, data_train_model=data_train_model,
                         data_test_model=data_test_model, data_train_monitor=data_train_monitor,
                         data_test_monitor=data_test_monitor, data_run=data_run, model_trainer=model_trainer,
                         model_name=model_name, model_path=model_path, n_epochs=n_epochs, batch_size=batch_size,
                         monitor_manager=monitor_manager, alphas=alphas)


    ##########
    history_run.update_statistics(1)
    fn = history_run.false_negatives()
    fp = history_run.false_positives()
    tp = history_run.true_positives()
    tn = history_run.true_negatives()
    novelty_results = novelty_wrapper_run.evaluate_detection(1)
    storage_1 = CoreStatistics(fn=fn, fp=fp, tp=tp, tn=tn, novelties_detected=len(novelty_results["detected"]),
                               novelties_undetected=len(novelty_results["undetected"]))
    storages_1 = [storage_1]

    
    history_run.update_statistics(2)
    fn = history_run.false_negatives()
    fp = history_run.false_positives()
    tp = history_run.true_positives()
    tn = history_run.true_negatives()
    novelty_results = novelty_wrapper_run.evaluate_detection(2)
    storage_2 = CoreStatistics(fn=fn, fp=fp, tp=tp, tn=tn, novelties_detected=len(novelty_results["detected"]),
                               novelties_undetected=len(novelty_results["undetected"]))
    storages_2 = [storage_2]

    storages_at = []

    for history_alpha, novelty_wrapper_alpha, alpha in zip(
            histories_alpha_thresholding, novelty_wrappers_alpha_thresholding, alphas):
        # history_alpha.update_statistics(0, confidence_threshold=alpha)  # not needed: history is already set
        fn = history_alpha.false_negatives()
        fp = history_alpha.false_positives()
        tp = history_alpha.true_positives()
        tn = history_alpha.true_negatives()
        novelty_results = novelty_wrapper_alpha.evaluate_detection(0)
        storage = CoreStatistics(fn=fn, fp=fp, tp=tp, tn=tn, novelties_detected=len(novelty_results["detected"]),
                                 novelties_undetected=len(novelty_results["undetected"]))
        storages = [storage]
        storages_at.append(storages)

    # store results
    store_core_statistics(storages_1, "monitor1")
    store_core_statistics(storages_2, "monitor2")
    store_core_statistics(storages_at, alphas)

    # load results
    storages_1b = load_core_statistics("monitor1")
    storages_2b = load_core_statistics("monitor2")
    storages_atb = load_core_statistics(alphas)
    pass
    


if __name__ == "__main__":
    run_script()
