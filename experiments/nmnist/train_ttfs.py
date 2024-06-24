from pathlib import Path
import cupy as cp
import numpy as np

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
 
from Dataset import Dataset
from bats.Monitors import *
from bats.Layers import InputLayer, LIFLayer
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset


# Dataset
DATASET_PATH = os.path.join('experiments', 'nmnist', 'data', '468j46mzdv-1')

N_INPUTS = 34*34*2
SIMULATION_TIME = 0.3

# Hidden layer
N_NEURONS_1 = 512
TAU_S_1 = 0.100
THRESHOLD_HAT_1 = 2.0
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 30

# Output_layer
N_OUTPUTS = 10
TAU_S_OUTPUT = 0.100
THRESHOLD_HAT_OUTPUT = 3.0
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 30

# Training parameters
N_TRAINING_EPOCHS = 15
N_TRAIN_SAMPLES = 6000
N_TEST_SAMPLES = 1000
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = 0.002
LR_DECAY_EPOCH = 1  # Perform decay very n epochs
LR_DECAY_FACTOR = 1.0
MIN_LEARNING_RATE = 0
TAU_LOSS = 0.06
TARGET_FALSE = 3
TARGET_TRUE = 15

# Plot parameters
EXPORT_METRICS = True
SAVE_DIR = Path("./best_model")


def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(0.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


def weight_initializer_out(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(0.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


if __name__ == "__main__":
    max_int = np.iinfo(np.int32).max
    np_seed = np.random.randint(low=0, high=max_int)
    cp_seed = np.random.randint(low=0, high=max_int)
    np.random.seed(np_seed)
    cp.random.seed(cp_seed)
    device = torch.device('cuda')

    print(f"Numpy seed: {np_seed}, Cupy seed: {cp_seed}")

    for DT in [0.01, 0.008, 0.006]:
        # Dataset
        SIMULATION_TIME = 0.3
        print("Loading datasets...")
        if DT < 0.001:
            dataset = Dataset(batch_size=TRAIN_BATCH_SIZE)
        else:     
            dataset = Dataset(batch_size=TRAIN_BATCH_SIZE, num_time_bins=int(SIMULATION_TIME/DT))
        EXPORT_DIR = Path(f"{DT} dt output_metrics 3.0")
        if not EXPORT_DIR.exists():
            os.makedirs(EXPORT_DIR)
        print("Creating network... for DT = ", DT)
        network = Network()

        input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
        network.add_layer(input_layer, input=True)

        hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                                theta=THRESHOLD_HAT_1,
                                delta_theta=DELTA_THRESHOLD_1,
                                time_delta=DT,
                                weight_initializer=weight_initializer,
                                max_n_spike=SPIKE_BUFFER_SIZE_1,
                                name="Hidden layer 1")
        network.add_layer(hidden_layer)

        output_layer = LIFLayer(previous_layer=hidden_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                                theta=THRESHOLD_HAT_OUTPUT,
                                delta_theta=DELTA_THRESHOLD_OUTPUT,
                                time_delta=DT,
                                weight_initializer=weight_initializer_out,
                                max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                                name="Output layer")
        network.add_layer(output_layer)

        loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
        # loss_fct = SpikeCountLoss()
        optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)

        # Metrics
        training_steps = 0
        # train_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_train", decimal=4)
        train_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_train")
        train_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
        train_silent_monitors = {l: SilentNeuronsMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
        train_time_monitor = TimeMonitor()
        all_train_monitors = [train_accuracy_monitor, train_time_monitor]
        all_train_monitors.extend(train_silent_monitors.values())
        all_train_monitors.extend(train_spike_counts_monitors.values())
        train_monitors_manager = MonitorsManager(all_train_monitors,
                                                print_prefix="Train | ")
        

        # test_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_test", decimal=4)
        test_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_test")
        test_learning_rate_monitor = ValueMonitor(name="Learning rate", decimal=5)
        # Only monitor LIF layers
        test_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
        test_silent_monitors = {l: SilentNeuronsMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
        test_norm_monitors = {l: WeightsNormMonitor(l.name, export_path=EXPORT_DIR / ("weight_norm_" + l.name))
                            for l in network.layers if isinstance(l, LIFLayer)}
        test_time_monitor = TimeMonitor()
        all_test_monitors = [test_accuracy_monitor, test_learning_rate_monitor]
        all_test_monitors.extend(test_spike_counts_monitors.values())
        all_test_monitors.extend(test_silent_monitors.values())
        all_test_monitors.extend(test_norm_monitors.values())
        all_test_monitors.append(test_time_monitor)
        test_monitors_manager = MonitorsManager(all_test_monitors,
                                                print_prefix="Test | ")

        best_acc = 0.0

        print("Training...")
        for epoch in range(N_TRAINING_EPOCHS):
            train_time_monitor.start()
            # Learning rate decay
            if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
                optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE)

            for batch_idx in range(N_TRAIN_BATCH):
                # Get next batch
                spikes, n_spikes, labels = dataset.get_train_batch()

                # Inference
                network.reset()
                network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME, training=True)
                out_spikes, n_out_spikes = network.output_spike_trains

                # Predictions, loss and errors
                pred = loss_fct.predict(out_spikes, n_out_spikes)
                loss, errors = loss_fct.compute_loss_and_errors(out_spikes, n_out_spikes, labels)

                pred_cpu = pred.get()
                loss_cpu = loss.get()
                n_out_spikes_cpu = n_out_spikes.get()

                # Update monitors
                # train_loss_monitor.add(loss_cpu)
                train_accuracy_monitor.add(pred, labels)
                # train_silent_label_monitor.add(n_out_spikes_cpu, labels)

                # Compute gradient
                gradient = network.backward(errors)
                avg_gradient = [None if g is None else cp.mean(g, axis=0) for g in gradient]
                del gradient

                # Apply step
                deltas = optimizer.step(avg_gradient)
                del avg_gradient

                network.apply_deltas(deltas)
                del deltas

                for l, mon in train_spike_counts_monitors.items():
                    mon.add(l.spike_trains[1])

                for l, mon in train_silent_monitors.items():
                    mon.add(l.spike_trains[1])

                training_steps += 1
                epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES
                    
            train_monitors_manager.record(epoch_metrics)
            train_monitors_manager.print(epoch_metrics)
            train_monitors_manager.export()

            # test_time_monitor.start()
            # for batch_idx in range(N_TEST_BATCH):
            #     spikes, n_spikes, labels = dataset.get_test_batch()
            #     network.reset()
            #     network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME)
            #     out_spikes, n_out_spikes = network.output_spike_trains

            #     pred = loss_fct.predict(out_spikes, n_out_spikes)
            #     loss = loss_fct.compute_loss(out_spikes, n_out_spikes, labels)

            #     # test_loss_monitor.add(loss_cpu)
            #     test_accuracy_monitor.add(pred, labels)

            #     for l, mon in test_spike_counts_monitors.items():
            #         mon.add(l.spike_trains[1])

            #     for l, mon in test_silent_monitors.items():
            #         mon.add(l.spike_trains[1])

            # for l, mon in test_norm_monitors.items():
            #     mon.add(l.weights)

            # test_learning_rate_monitor.add(optimizer.learning_rate)

            # records = test_monitors_manager.record(epoch_metrics)
            # test_monitors_manager.print(epoch_metrics)
            # test_monitors_manager.export()

            # acc = records[test_accuracy_monitor]
            # if acc > best_acc:
            #     best_acc = acc
            #     network.store(SAVE_DIR)
            #     print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save to: {SAVE_DIR}")
        
        cr = train_accuracy_monitor.convergence_rate()
        path_save = os.path.join(EXPORT_DIR, 'coverge.txt')
        with open(path_save, 'a') as f:
            f.write(str(DT) + ' ' + str(cr) + '\n')
