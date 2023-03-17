import sampler
import numpy as np
import mlp
from bridges.dataset import Dataset
from bridges import trainer
import torch
import matplotlib.pyplot as plt
import scipy as sp
from bridges.brownianBridge import MixtureBrownianBridges
from collections import OrderedDict
from bridges.utils import get_dataset

SIZE_training = 100000
SIZE_test = 1000
batch_size = 1000
def likelihood(mu1, mu2):
    mu = mu1
    if np.random.uniform() < 0.5:
        mu = mu2

    return np.random.normal(size=(1,)) + mu

def compute_likelihood(theta, mu1, mu2):
    log_first_mode = -1/2*(mu1 - theta)**2 - 1/2 * np.log(2*np.pi) - 1/2 * np.log(1)
    log_second_mode = -1/2*(mu2 - theta)**2/1 - 1/2 * np.log(2*np.pi) - 1/2 * np.log(1)
    return 0.5*np.exp(log_first_mode) + 0.5*np.exp(log_second_mode)


def compute_partition_function(mu1, mu2):
    first_mode = 1/40 * (sp.stats.norm.cdf(10, loc=mu1, scale=1) - sp.stats.norm.cdf(-10, loc=mu1, scale=1))
    second_mode = 1/40 * (sp.stats.norm.cdf(10, loc=mu2, scale=np.sqrt(1)) - sp.stats.norm.cdf(-10, loc=mu2, scale=np.sqrt(1)))
    return first_mode + second_mode


#distributions = {"distrib0":{"t":0, "mu1":0, "mu2":0},"distrib1":{"t":0.5, "mu1":-5, "mu2":-5,"steps":500},
#                 "distrib2":{"t":1, "mu1":-10, "mu2":-10, "steps":500}}

distributions = {"distrib0":{"t":0, "mu1":0, "mu2":0},"distrib1":{"t":1, "mu1":-2, "mu2":-2,"steps":1000}}
datasets = {distrib:np.array([likelihood(params["mu1"], params["mu2"]) for _ in range(SIZE_training)])
            for distrib, params in distributions.items()}
datasets["sampled_times"] = np.random.uniform(size=SIZE_training)

final_dataset = get_dataset(distributions, datasets)
brownianBridge = MixtureBrownianBridges()
sampled_x_t = []
for i in range(SIZE_training):
    if i%1000 == 0:
        print(i)

    x_0 = final_dataset["starting_point"][i]
    x_T = final_dataset["ending_point"][i]
    sampled_time_difference = final_dataset["sampled_time_difference"][i]
    time_between_distrib = final_dataset["time_between_distrib"][i]
    x_t = brownianBridge.sample_marginal(x_0=x_0, x_T=x_T, t=sampled_time_difference,
                                         T=time_between_distrib, n_sample=1)
    sampled_x_t.append(x_t)

sampled_x_t = np.array(sampled_x_t)
final_dataset["sampled_x_t"] = sampled_x_t

dataset_test = {distrib:np.array([likelihood(params["mu1"], params["mu2"]) for _ in range(SIZE_test)])
            for distrib, params in distributions.items()}

dataset_test["sampled_times"] = np.random.uniform(size=SIZE_test)

final_dataset_test = get_dataset(distributions, dataset_test)
print(final_dataset["distribution_number"])
print("\n")
sampled_x_t_test = []
for i in range(SIZE_test):
    if i%1000 == 0:
        print(i)

    x_0 = final_dataset_test["starting_point"][i]
    x_T = final_dataset_test["ending_point"][i]
    sampled_time_difference = final_dataset_test["sampled_time_difference"][i]
    time_between_distrib = final_dataset_test["time_between_distrib"][i]
    x_t = brownianBridge.sample_marginal(x_0=x_0, x_T=x_T, t=sampled_time_difference,
                                         T=time_between_distrib, n_sample=1)
    sampled_x_t_test.append(x_t)

sampled_x_t_test = np.array(sampled_x_t_test)
final_dataset_test["sampled_x_t"] = sampled_x_t_test

#hist(dataset_sampled[:, 0], bins=40)
#plt.show()
perceptron = mlp.MLP("mlp_mixture.yaml")
train = trainer.Trainer(final_dataset, final_dataset_test)
all_losses, losses_last = train.train(perceptron, batch_size=batch_size,  n_epochs=50, lr=0.0003)

#plt.plot(all_losses)
#plt.show()
#discretization_times = torch.tensor(np.linspace(0, 1, 1000)[1:], dtype=torch.float32)[:, None, None]
sampled_posterior = {"distrib"+str(i):[] for i in range(1, 3)}
x_0 = likelihood(distributions["distrib0"]["mu1"], distributions["distrib0"]["mu2"])
x_T = torch.ones((1,1))*x_0
for i in range(1000):
    print(i)
    for n_distrib in range(1, 2):
        steps = distributions["distrib"+str(n_distrib)]["steps"]
        time_horizon = distributions["distrib"+str(n_distrib)]["t"] - distributions["distrib"+str(n_distrib-1)]["t"]
        discretization_times = torch.tensor(np.linspace(0, time_horizon, steps), dtype=torch.float32)[:, None, None]
        traj, x_T = brownianBridge.euler_maruyama(x_0=x_T, times=discretization_times, tau=time_horizon,
                                            distrib_number=torch.ones((1,1), dtype=torch.float32)*float(n_distrib),
                                                     network=perceptron)

        sampled_posterior["distrib"+str(n_distrib)].append(x_T.detach().numpy())

for n_distrib in range(1, 2):
    sampled_posterior["distrib"+str(n_distrib)] = np.array(sampled_posterior["distrib"+str(n_distrib)])

print(sampled_posterior)

#low = np.min(sampled_posterior)
#high = np.max(sampled_posterior)

#low = np.min(sorted_params[:10000, 0])
#low = - 5
#high = np.max(sorted_params[:10000, 0])
#high = 5
#xx = np.linspace(low, high, 10000)
#y = np.array([compute_prior(x)*compute_likelihood(x, observed_data)/compute_partition_function(observed_data) for x in xx])
#y = np.array([compute_likelihood(x, observed_data) for x in xx])
plt.hist(sampled_posterior["distrib1"][:, -1, 0], bins=40, density=True, alpha=0.5)
#plt.hist(sampled_posterior["distrib2"][:, -1, 0], bins=40, density=True, alpha=0.5)
#plt.hist(sorted_params[:10000, 0], bins=40, density=True, alpha=0.5)
#plt.plot(xx, y)
plt.show()
