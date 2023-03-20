import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset


class Dataset(torchDataset):
    def __init__(self, data, perturbed_data=None, sampled_times=None, distrib_number=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sampled_times = None
        self.distrib_number = None
        if perturbed_data is not None:
            self.perturbed_data = torch.tensor(data, dtype=torch.float32)
            self.sampled_times = torch.tensor(sampled_times, dtype=torch.float32)
            if distrib_number is not None:
                self.distrib_number = torch.tensor(distrib_number, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.perturbed_data is None:
            return self.data[idx]

        if self.distrib_number is None:
            return self.data[idx], self.perturbed_data[idx], self.sampled_times[idx]

        return self.data[idx], self.perturbed_data[idx], self.sampled_times[idx], self.distrib_number[idx]

