from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import scipy


class AbstractMonitor(ABC):
    def __init__(self, name: str, export_path: Optional[Path] = None, decimal: int = 2):
        self._name: str = name
        self._export_path: Optional[Path] = export_path
        self._decimal = decimal

        self._epochs = []
        self._values = []

    def __str__(self) -> str:
        return self._name + ": " + self._format()

    def _format(self) -> str:
        return str(np.around(self._values[-1], self._decimal))

    def reset(self) -> None:
        self._epochs = []
        self._values = []

    def _record(self, epoch: float, value: float) -> None:
        self._epochs.append(epoch)
        self._values.append(value)

    @abstractmethod
    def record(self, epoch: float) -> float:
        pass

    def export(self) -> None:
        if self._export_path is None:
            return
        epochs = np.array(self._epochs)

        if isinstance(self._values, list):
            values = np.array(self._values)
        else:
            values = cp.asnumpy(self._values)
        np.savez(self._export_path, epochs=epochs, values=values)

        plt.plot(self._epochs, self._values)
        plt.xlabel("Epoch")
        plt.ylabel(self._name)
        plt.savefig(self._export_path.with_suffix('.png'))
        plt.close()

    def convergence_rate(self) -> float:
        convergence_rate = scipy.trapz((self._values), dx=1)
        return convergence_rate