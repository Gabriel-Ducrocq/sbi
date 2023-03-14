import sampler
import numpy as np
import mlp
from bridges.dataset import Dataset
from bridges import trainer
import torch
import matplotlib.pyplot as plt
import scipy as sp
from bridges.brownianBridge import MixtureBrownianBridges


SIZE_training = 100000
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





mu1 = -10
mu2 = 10
Sampler = sampler.Sampler(100000)
dataset_sampled = np.array([likelihood(mu1, mu2) for _ in range(SIZE_training)])
sampled_times = np.random.uniform(size=SIZE_training)
brownianBridge = MixtureBrownianBridges(sigma=5)
sampled_x_t = []
i = 1
for param, t in zip(dataset_sampled, sampled_times):
    if i%1000 == 0:
        print(i)

    i += 1
    x_0 = np.random.normal(size=(1,1))
    x_t = brownianBridge.sample_marginal(x_0=x_0, x_T=param, t=t, T=1, n_sample=1)
    sampled_x_t.append(x_t)

sampled_x_t = np.array(sampled_x_t)
print("SHAPE")
print(sampled_times.shape)
data_train = Dataset(dataset_sampled, perturbed_data=sampled_x_t[:, 0], sampled_times=sampled_times[:, None])



Sampler_test = sampler.Sampler(1000)
dataset_sampled_test = np.array([likelihood(mu1, mu2) for _ in range(1000)])
sampled_times_test = np.random.uniform(size=1000)
sampled_x_t_test = []
i = 1
for param, t in zip(dataset_sampled_test, sampled_times_test):
    if i%10 == 0:
        print(i)

    i += 1
    x_0 = np.random.normal(size=(1,1))
    x_t = brownianBridge.sample_marginal(x_0=x_0, x_T=param, t=t, T=1, n_sample=1)
    sampled_x_t_test.append(x_t)


sampled_x_t_test = np.array(sampled_x_t_test)
data_test = Dataset(dataset_sampled_test, perturbed_data=sampled_x_t_test[:, 0],
                             sampled_times=sampled_times_test[:, None])

#hist(dataset_sampled[:, 0], bins=40)
#plt.show()
perceptron = mlp.MLP("mlp_mixture.yaml")
#result = perceptron.forward(dataset.parameters)
train = trainer.Trainer(data_train, data_test)
all_losses, losses_last = train.train(perceptron, batch_size=10,  n_epochs =100, lr=0.0003)
times = data_test.sampled_times

#plt.plot(all_losses)
#plt.show()
discretization_times = torch.tensor(np.linspace(0, 1, 1000)[1:], dtype=torch.float32)[:, None, None]
sampled_posterior = []

for i in range(1000):
    print("Iteration:", i)
    x_0 = torch.randn((1, 1), dtype=torch.float32)
    sampled_post = brownianBridge.euler_maruyama(x_0=x_0, times=discretization_times, tau=1, network=perceptron)
    sampled_posterior.append(sampled_post)

sampled_posterior = np.array(sampled_posterior)

low = np.min(sampled_posterior)
high = np.max(sampled_posterior)

#low = np.min(sorted_params[:10000, 0])
#low = - 5
#high = np.max(sorted_params[:10000, 0])
#high = 5
xx = np.linspace(low, high, 10000)
#y = np.array([compute_prior(x)*compute_likelihood(x, observed_data)/compute_partition_function(observed_data) for x in xx])
#y = np.array([compute_likelihood(x, observed_data) for x in xx])
plt.hist(sampled_posterior[:, -1, 0], bins=40, density=True, alpha=0.5)
#plt.hist(sorted_params[:10000, 0], bins=40, density=True, alpha=0.5)
#plt.plot(xx, y)
plt.show()
