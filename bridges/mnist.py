import numpy as np
import torch
#import matplotlib.pyplot as plt
#import scipy as sp
from bridges.brownianBridge import MixtureBrownianBridges
from collections import OrderedDict
from bridges.utils import get_dataset
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import mlp
import trainer
from bridges.dataset import Dataset

def sample_perturbed_data(x0, xT, size, sampled_times, T=1):
    perturbed_samples = torch.empty((SIZE_training, 28 * 28))
    for i in range(size):
        if i % 1000 == 0:
            print(i)

        x_0 = x0[i]
        x_T = xT[i]
        sampled_time_difference = sampled_times[i, :]
        time_between_distrib = T
        x_t = brownianBridge.sample_marginal(x_0=x_0, x_T=x_T, t=sampled_time_difference,
                                             T=time_between_distrib, n_sample=1)

        perturbed_samples[i, :] = x_t

    return perturbed_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
SIZE_training = 60000
SIZE_test = 10000
batch_size = 600
mnist_dataset = torchvision.datasets.MNIST("../datasets/", download=True, transform= transforms.ToTensor())
full_data_loader = DataLoader(mnist_dataset, batch_size=SIZE_training, shuffle=True)
full_data_load_iter = iter(full_data_loader)
full_dataset, _ = next(full_data_load_iter)
full_dataset = torch.flatten(full_dataset[:, 0, :, :], start_dim=-2, end_dim=-1)

mnist_dataset_test = torchvision.datasets.MNIST("../datasets/", download=True, transform= transforms.ToTensor(), train=False)
full_data_loader_test = DataLoader(mnist_dataset_test, batch_size=10000, shuffle=False)
full_data_load_test_iter = iter(full_data_loader_test)
full_dataset_test, _ = next(full_data_load_test_iter)
full_dataset_test = torch.flatten(full_dataset_test[:, 0, :, :], start_dim=-2, end_dim=-1)

brownianBridge = MixtureBrownianBridges()


sampled_times = torch.rand((SIZE_training, 1))
x0 = torch.zeros((SIZE_training, 28*28))
perturbed_samples = sample_perturbed_data(x0, full_dataset, SIZE_training, sampled_times, 1)
final_dataset = Dataset(full_dataset, perturbed_samples, sampled_times)

sampled_times_test = torch.rand((SIZE_test, 1))
x0_test = torch.zeros((SIZE_test, 28*28))
perturbed_samples_test = sample_perturbed_data(x0_test, full_dataset_test, SIZE_test, sampled_times_test, 1)
final_dataset_test = Dataset(full_dataset_test, perturbed_samples_test, sampled_times_test)

#argmin = torch.argmin(sampled_times)
#plt.imshow(torch.reshape(perturbed_samples[argmin, :], (28, 28)), cmap="gray")
#plt.show()

#argmax = torch.argmax(sampled_times)
#plt.imshow(torch.reshape(perturbed_samples[argmax, :], (28, 28)), cmap="gray")
#plt.show()
perceptron = mlp.MLP("../mlp_mixture.yaml")
train = trainer.Trainer(final_dataset, final_dataset_test)
all_losses, losses_last = train.train(perceptron, batch_size=batch_size,  n_epochs=100, lr=0.0003)

brownianBridge = MixtureBrownianBridges()
network = torch.load("full_model")
time_grid = torch.tensor(np.linspace(0, 1, 1000)[1:], dtype=torch.float32)[:, None, None]
all_samples = []
for i in range(10):
    print(i)
    traj, sample = brownianBridge.euler_maruyama(torch.zeros((1, 28*28)), time_grid, 1, network)
    all_samples.append(sample)

for i in range(10):
    samp = all_samples[i]
    #plt.imshow(torch.reshape(samp, (28, 28)).detach().numpy(), cmap="gray")
    #plt.show()