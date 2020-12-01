import sys
sys.path.insert(0, '/home/nikhil/Downloads/pavt/Outside-the-Box')

from data import *
from utils import *
from abstractions import *
from trainers import *
from monitoring import *

def run_script():
    # options
    seed = 0
    data_name = "ToyData"
    classes = [0, 1]
    n_classes = 2
    data_train_model = DataSpec(classes=classes)
    data_test_model = DataSpec(classes=classes)
    data_train_monitor = DataSpec(classes=classes)
    data_test_monitor = DataSpec(classes=classes)
    data_run = DataSpec(classes=classes)
    model_name = "ToyModel"
    model_path = "Toy-model.h5"
    n_epochs = 0
    batch_size = 128
    model_trainer = StandardTrainer()

    all_classes_network, labels_network, all_classes_rest, labels_rest = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)
    model, _ = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model,
                         n_classes=n_classes, model_trainer=model_trainer, n_epochs=n_epochs, batch_size=batch_size,
                         statistics=Statistics(), model_path=model_path)
    
    print("\nmodels ---> ")
    print(model)
    print(model.weight_matrizes)
    print(model.activations)
    #print(model.layers.output_shape)

    # create monitor
    #layer2abstraction = {1: BoxAbstraction(euclidean_distance)}
    layer2abstraction = {1: StarAbstraction(euclidean_distance)}
    print("\nlayer2abstraction ---> ")
    print(layer2abstraction[1].sets)
    print(layer2abstraction[1].dim)
    print(layer2abstraction[1].epsilon)
    print(layer2abstraction[1].epsilon_relative)
    print(layer2abstraction[1].confidence_fun)
    monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor_manager = MonitorManager([monitor], n_clusters=1)
    #for m in monitor_manager._monitors:
    #    print(m._layer2abstraction[1].dim)
    print("\nmonitor manager ---> ")
    #print(monitor_manager._layers)

    # run instance
    monitor_manager.normalize_and_initialize(model, len(labels_rest))
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics(), ignore_misclassifications=False)

    print(monitor_manager)
    print(monitor_manager._layers)
    print("\nmonitors ---> ")
    box1=monitor_manager._monitors[0]._layer2abstraction[1]._abstractions[0].sets[0]
    """
    print("Converting box to Star")
    star1=box1.toStar()
    print(star1)
    star2=star1.scalarMap(0.7)
    print(star2)
    star3=Star.concatenateStars([star1,star1])
    print("Finished")
    """
    for m in monitor_manager._monitors:
        print(m._layer2abstraction[1]._abstractions)
        print(m._layer2abstraction[1]._abstractions[0].sets[0])
        print(m._layer2abstraction[1]._abstractions[0].sets[0])
        s1=m._layer2abstraction[1]._abstractions[0].sets[0]
        s0=m._layer2abstraction[1]._abstractions[1].sets[0]
        #s0.plot()
        #s1.plot()
    Star.plot([s0,s1])
    #print(s1.v)
    #print(s1.c)
    #print(s1.d)
    #print(s0.v)
    #print(s0.c)
    #print(s0.d)
    """
    from matplotlib import pyplot as plt
    #fig = plt.figure(figsize =(8,8))
    plt.plot(vert)
    plt.show()
    #plt.savefig('an2.png')
    """
    
    ### adding code 

    



    # create plots
    history = History()
    history.set_ground_truths(data_run.ground_truths())
    layer = 1
    layer2values, _ = obtain_predictions(model=model, data=data_run, layers=[layer])
    #print(layer2values)
    history.set_layer2values(layer2values)
    plot_2d_projection(history=history, monitor=None, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[0, 1])
    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[0, 1])

    save_all_figures(close=True)


if __name__ == "__main__":
    run_script()
