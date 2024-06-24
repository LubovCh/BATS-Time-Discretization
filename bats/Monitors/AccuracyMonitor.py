from bats.AbstractMonitor import AbstractMonitor
import numpy as np
import cupy as cp

class AccuracyMonitor(AbstractMonitor):
    def __init__(self, **kwargs):
        super().__init__("Accuracy (%)", **kwargs)
        self._hits = 0
        self._n_samples = 0

    def add(self, predictions: cp.ndarray, targets: cp.ndarray) -> None:
        self._hits += cp.sum(predictions == targets)
        self._n_samples += targets.shape[0]
        

    def record(self, epoch) -> float:
        accuracy = self._hits / self._n_samples 
        super()._record(epoch, accuracy)
        self._hits = 0
        self._n_samples = 0
        return accuracy
