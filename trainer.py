import torch
import numpy as np
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, dataset, dataset_test):
        self.dataset = dataset
        self.dataset_test = dataset_test
        self.size_data_set = len(dataset)
        self.size_data_set_test = len(dataset_test)

    def compute_msd(self, predicted, label):
        return torch.mean(torch.sum((predicted - label)**2, dim=-1))

    def train(self, network, batch_size, n_epochs):
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
        all_losses = []
        all_losses_test = []
        assert self.size_data_set % batch_size == 0, "Batch size not a divider of dataset size"
        for epoch in range(n_epochs):
            print("Epoch:", epoch)
            data_loader = iter(DataLoader(self.dataset, batch_size=batch_size, shuffle=True))
            data_loader_test = iter(DataLoader(self.dataset_test, batch_size=self.size_data_set_test, shuffle=False))
            for i in range(int(self.size_data_set/batch_size)):
                batch_parameters, batch_data, batch_perturbed_parameters, batch_times = next(data_loader)
                x = torch.concat([batch_data, batch_perturbed_parameters[:, 0, :], batch_times[:, None]], dim=-1)
                predicted_params = network.forward(x)
                loss = self.compute_msd(predicted_params, batch_parameters)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print(loss)
                all_losses.append(loss.detach().numpy())
                #print(predicted_params[0, :])
                #print(batch_perturbed_parameters[0, :])
                #print("\n")

            batch_parameters, batch_data, batch_perturbed_parameters, batch_times = next(data_loader_test)
            x = torch.concat([batch_data, batch_perturbed_parameters[:, 0, :], batch_times[:, None]], dim=-1)
            predicted_params = network.forward(x)
            loss_test = self.compute_msd(predicted_params, batch_parameters)
            all_losses_test.append(loss_test.detach().numpy())
            print("loss test:", loss_test)

        return np.array(all_losses_test), torch.sum((predicted_params - batch_parameters)**2,dim =-1)


