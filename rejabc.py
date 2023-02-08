import numpy as np

class REJABC():
    def __init__(self, dataset, observed_data):
        self.dataset = dataset
        self.observed_data = observed_data

    def sample(self, compute_distances):
        params = self.dataset.parameters
        data = self.dataset.data
        distances = compute_distances(data.detach().numpy(), self.observed_data)
        sorted_by_distance = sorted(zip(distances, params.detach().numpy()))
        return tuple(list(zip(*sorted_by_distance)))

