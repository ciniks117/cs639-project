
import sys
sys.path.insert(0, '/home/nikhil/Downloads/pavt/Outside-the-Box')
from data import *
from run.experiment_helper import *

import tensorflow_model_optimization as tfmot

def run_script():
    model_name, data_name, stored_network_name, total_classes = instance_MNIST()
    classes = [0, 1]
    n_classes = 2
    classes_string = classes2string(classes)
    model_path = "{}_{}.h5".format(stored_network_name, classes_string)
    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    data_train_monitor = DataSpec(randomize=False, classes=classes)
    data_test_monitor = DataSpec(randomize=False, classes=classes)
    data_run = DataSpec(randomize=False, classes=[0, 1, 2])

    all_classes_network, labels_network, all_classes_rest, labels_rest = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)
    model, _ = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model,
                         n_classes=n_classes, model_trainer=None, n_epochs=None, batch_size=None, statistics=None,
                         model_path=model_path)
    
    print(model)
    #print(model.get_weights())
    print(len(model.layers))
    print(model.layers[0])
    print(model.layers[1])
    print(model.layers[2])
    print(model.layers[3])
    #for layer in model.layers:
    #    weights = layer.get_weights()
    #    print(weights)
    #    print("###################")
    # use this fcn : model.set_weights(weights)

    

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 128
    epochs = 2
    validation_split = 0.1 # 10% of training set will be used for validation set.

    #num_images = train_images.shape[0] * (1 - validation_split)
    #end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
    end_step = 800

    print(" END STEP : "+str(end_step))
    # Define model for pruning.
    pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model_for_pruning.summary()

    logdir = tempfile.mkdtemp()

    callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

    _,  model_for_pruning_accuracy = model_for_pruning.evaluate(
       test_images, test_labels, verbose=0)
    
    print('Baseline test accuracy:', baseline_model_accuracy)
    print('Pruned test accuracy:', model_for_pruning_accuracy)





    # create monitor
    layer2abstraction = {-2: BoxAbstraction(euclidean_distance)}
    print(layer2abstraction[-2])
    print(layer2abstraction[-2].sets)
    print(layer2abstraction[-2].dim)
    monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor_manager = MonitorManager([monitor], n_clusters=3)

    # run instance
    monitor_manager.normalize_and_initialize(model, len(labels_rest))
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics())
    print(monitor_manager)
    print(monitor_manager._layers)
    #for m in monitor_manager._monitors:
    #    print(m._layer2abstraction[8]._abstractions)

    # create plot
    history = History()
    history.set_ground_truths(data_run.ground_truths())
    layer = 8
    layer2values, _ = obtain_predictions(model=model, data=data_run, layers=[layer])
    history.set_layer2values(layer2values)
    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[3, 13])

    save_all_figures()


if __name__ == "__main__":
    print(tf.__version__)
    run_script()
