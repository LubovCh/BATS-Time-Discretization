from pathlib import Path
from matplotlib import animation, pyplot as plt
import numpy as np
import cupy as cp
import os
import glob
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import tensorflow as tf
from tensorflow import Tensor

TIME_WINDOW = 100e-3
MAX_VALUE = 255.0

class Dataset:
    def __init__(self, batch_size = None, num_time_bins = None) -> None:
        sensor_size = tonic.datasets.NMNIST.sensor_size

        if num_time_bins is None:
            frame_transform = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size,
                                                            time_window=0.3),
                                        ])
        else: 
            frame_transform = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size,
                                                            n_time_bins=num_time_bins),
                                        ])
        trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
        testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

        cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
        cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')
        
        self.trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
        self.testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)


    def get_train_batch(self):
        events, labels = next(iter(self.trainloader))

        labels = labels.cpu().numpy()
        labels = cp.asarray(labels)

        spikes_per_neuron, n_spikes_per_neuron = self.__to_spikes(events)
        return spikes_per_neuron, n_spikes_per_neuron, labels
        
    

    def get_test_batch(self):

        events, labels = next(iter(self.testloader))

        labels = labels.cpu().numpy()
        labels = cp.asarray(labels)

        spikes_per_neuron, n_spikes_per_neuron = self.__to_spikes(events)
        return spikes_per_neuron, n_spikes_per_neuron, labels

    def __to_spikes(self, samples):

        samples = samples.cpu().numpy()

        spike_times = samples.reshape((samples.shape[0], 34*34*2, samples.shape[1]))
        spike_times = TIME_WINDOW * (1.0 - (spike_times / MAX_VALUE))
        spike_times[spike_times == TIME_WINDOW] = np.inf
        inf_counts = np.count_nonzero(spike_times != np.inf, axis=2)
        n_spike_per_neuron = np.isfinite(spike_times).astype('int').reshape((samples.shape[0], 34*34*2, samples.shape[1]))
        return spike_times, inf_counts


       