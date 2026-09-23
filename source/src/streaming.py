from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .models import ProbeDataset


@dataclass
class SampleBatch:
    steps: np.ndarray
    time_s: np.ndarray
    values: np.ndarray


class StreamingSource(Protocol):
    def reset(self) -> None: ...
    def read(self, maximum_samples: int) -> SampleBatch: ...


class CsvReplaySource:
    def __init__(self, dataset: ProbeDataset) -> None:
        self.dataset = dataset
        self.reset()

    def reset(self) -> None:
        self.cursor = 0

    def read(self, maximum_samples: int) -> SampleBatch:
        if maximum_samples < 1:
            raise ValueError("每批读取数量必须大于 0。")
        end = min(self.cursor + int(maximum_samples), self.dataset.sample_count)
        selection = slice(self.cursor, end)
        batch = SampleBatch(
            steps=self.dataset.steps[selection].copy(),
            time_s=self.dataset.time_s[selection].copy(),
            values=self.dataset.values[selection].copy(),
        )
        self.cursor = end
        return batch
