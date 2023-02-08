import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset


class Dataset(torchDataset):
    def __init__(self, parameters, data, perturbed_params=None, sampled_times = None):
        self.parameters = torch.tensor(parameters, dtype=torch.float32)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.perturbed_params = None
        self.sampled_times = None
        if perturbed_params is not None:
            self.perturbed_params = torch.tensor(perturbed_params, dtype=torch.float32)
            self.sampled_times = torch.tensor(sampled_times, dtype=torch.float32)

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        if self.perturbed_params is None:
            return self.parameters[idx], self.data[idx]

        return self.parameters[idx], self.data[idx], self.perturbed_params[idx], self.sampled_times[idx]