import torch
import numpy as np
from bridges.dataset import Dataset
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, dataset, dataset_test, size_train=60000, size_test=10000):
        self.dataset = dataset
        self.dataset_test = dataset_test
        self.size_data_set = size_train
        self.size_data_set_test = size_test
        #self.complete_dataset_train = self.build_dataset(self.dataset)
        #self.complete_dataset_test = self.build_dataset(self.dataset_test)

    def build_dataset(self, dataset):
        ending_points = dataset["ending_point"]
        time_diff = dataset["sampled_time_difference"]
        distrib_number = dataset['distribution_number']
        sampled_x_t = dataset['sampled_x_t']
        return Dataset(ending_points[:, None], sampled_x_t[:, None], time_diff[:, None], distrib_number[:, None])


    def compute_msd(self, predicted, label):
        return torch.mean(torch.sum((predicted - label)**2, dim=-1))

    def train(self, network, batch_size, n_epochs, lr=0.0003):
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        all_losses_test = []
        assert self.size_data_set % batch_size == 0, "Batch size not a divider of dataset size"
        data_loader_test = iter(DataLoader(self.dataset, batch_size=self.size_data_set_test, shuffle=False))

        #batch_data, batch_perturbed_data, batch_times, batch_distrib_number = next(data_loader_test)
        batch_data, batch_perturbed_data, batch_times = next(data_loader_test)
        #x = torch.concat([batch_perturbed_data, batch_times, batch_distrib_number], dim=-1)
        x = torch.concat([batch_perturbed_data, batch_times], dim=-1)
        predicted_data = network.forward(x)
        loss_test = self.compute_msd(predicted_data, batch_data)

        all_losses_test.append(loss_test.detach().numpy())
        print("loss test:", loss_test)
        for epoch in range(n_epochs):
            all_losses = []
            print("Epoch:", epoch)
            data_loader = iter(DataLoader(self.dataset, batch_size=batch_size, shuffle=True))
            data_loader_test = iter(DataLoader(self.dataset_test, batch_size=self.size_data_set_test, shuffle=False))
            data_loader_train_test = iter(DataLoader(self.dataset, batch_size=100000, shuffle=False))
            for i in range(int(self.size_data_set/batch_size)):
                #batch_data, batch_perturbed_data, batch_times, batch_distrib_number = next(data_loader)
                batch_data, batch_perturbed_data, batch_times = next(data_loader)
                #print("train_data", batch_data)
                #x = torch.concat([batch_perturbed_data, batch_times, batch_distrib_number], dim=-1)
                x = torch.concat([batch_perturbed_data, batch_times], dim=-1)
                predicted_data = network.forward(x)
                loss = self.compute_msd(predicted_data, batch_data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print(loss)
                all_losses.append(loss.detach().numpy())
                #print(predicted_params[0, :])
                #print(batch_perturbed_parameters[0, :])
                #print("\n")

            #batch_data_test, batch_perturbed_data_test, batch_times_test, batch_distrib_number_test = next(data_loader_test)
            batch_data_test, batch_perturbed_data_test, batch_times_test = next(data_loader_test)
            #x = torch.concat([batch_perturbed_data_test, batch_times_test, batch_distrib_number_test], dim=-1)
            x = torch.concat([batch_perturbed_data_test, batch_times_test], dim=-1)
            predicted_data_test = network.forward(x)
            loss_test = self.compute_msd(predicted_data_test, batch_data_test)
            torch.save(network, "full_model")
            """
            batch_data_train_test, batch_perturbed_data_train_test, batch_times_train_test, batch_distrib_number_train_test = next(data_loader_train_test)
            x = torch.concat([batch_perturbed_data_train_test, batch_times_train_test, batch_distrib_number_train_test], dim=-1)
            predicted_data_train_test = network.forward(x)
            loss_train_test = self.compute_msd(predicted_data_train_test, batch_data_train_test)
            """
            all_losses_test.append(loss_test.detach().numpy())
            print("loss training:", np.mean(all_losses))
            print("loss test:", loss_test)
            #print("loss train test", loss_train_test)

        return np.array(all_losses_test), torch.sum((predicted_data - batch_data)**2,dim =-1)

