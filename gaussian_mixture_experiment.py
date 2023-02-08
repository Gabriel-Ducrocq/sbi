import sampler
import numpy as np
import mlp
from diffusion import Diffusion
import dataset
import trainer
import torch
import matplotlib.pyplot as plt
from rejabc import REJABC
import scipy as sp

def abc_distance(x, y):
    return np.sum((x - y)**2, axis= 1)

def prior():
    return np.random.uniform(-10, 10, size=(1,))

def likelihood(theta):
    sigma = 1
    if np.random.uniform() < 0.5:
        sigma = np.sqrt(0.01)

    return theta + np.random.normal(size=(1,))*sigma


def compute_prior(theta):
    if theta > - 10 and theta < 10:
        return 1/20

    return 0

def compute_likelihood(theta, observed_data):
    log_first_mode = -1/2*(observed_data - theta) - 1/2 * np.log(2*np.pi) - 1/2 * np.log(1)
    log_second_mode = -1/2*(observed_data - theta)/0.01 - 1/2 * np.log(2*np.pi) - 1/2 * np.log(0.01)
    return 0.5*np.exp(log_first_mode) + 0.5*np.exp(log_second_mode)


def compute_partition_function(observed_data):
    first_mode = 1/40 * (sp.stats.norm.cdf(10, loc=observed_data, scale=1) - sp.stats.norm.cdf(-10, loc=observed_data, scale=1))
    second_mode = 1/40 * (sp.stats.norm.cdf(10, loc=observed_data, scale=np.sqrt(0.01)) - sp.stats.norm.cdf(-10, loc=observed_data, scale=np.sqrt(0.01)))
    return first_mode + second_mode


def beta_t(t):
    return 1

def alpha_t(t):
    return 1

def b_t(t):
    return t


Sampler = sampler.Sampler(100000)
dataset_sampled = Sampler.sample(prior, likelihood)
sampled_times = np.random.uniform(size=100000)
diff = Diffusion(np.eye(1), b_t, alpha_t, beta_t)
sampled_x_t = []
i = 1
for param, t in zip(dataset_sampled.parameters, sampled_times):
    if i%1000 == 0:
        print(i)

    i += 1
    x_0 = np.random.normal(size=(1,1))
    x_t = diff.sample_marginal(x_0=x_0, x_tau=param[:, None].detach().numpy(), t=t, tau=1, n_sample=1)
    sampled_x_t.append(x_t)

sampled_x_t = np.array(sampled_x_t)
print(sampled_x_t.shape)
data_train = dataset.Dataset(dataset_sampled.parameters, dataset_sampled.data, perturbed_params=sampled_x_t, sampled_times=sampled_times)


Sampler_test = sampler.Sampler(1000)
dataset_sampled_test = Sampler_test.sample(prior, likelihood)
sampled_times_test = np.random.uniform(size=1000)
sampled_x_t_test = []
i = 1
for param, t in zip(dataset_sampled_test.parameters, sampled_times_test):
    if i%1000 == 0:
        print(i)

    i += 1
    x_0 = np.random.normal(size=(1,1))
    x_t = diff.sample_marginal(x_0=x_0, x_tau=param[:, None].detach().numpy(), t=t, tau=1, n_sample=1)
    sampled_x_t_test.append(x_t)


sampled_x_t = np.array(sampled_x_t_test)
data_test = dataset.Dataset(dataset_sampled_test.parameters, dataset_sampled_test.data, perturbed_params=sampled_x_t_test,
                             sampled_times=sampled_times_test)


true_param = prior()
observed_data = likelihood(true_param)



abc = REJABC(data_train, observed_data)
sorted_distances, sorted_params = abc.sample(compute_distances=abc_distance)
sorted_params = np.array(sorted_params)
print(sorted_params.shape)
#plt.hist(sorted_params[:100000, 0], density=True, alpha=0.5)
#plt.hist(sorted_params[:10000, 0], density=True, alpha=0.5)
#plt.show()

perceptron = mlp.MLP("mlp_mixture.yaml")
#result = perceptron.forward(dataset.parameters)
train = trainer.Trainer(data_train, data_test)
all_losses, losses_last = train.train(perceptron, batch_size=1000,  n_epochs =500)
times = data_test.sampled_times

discretization_times = torch.tensor(np.linspace(0, 1, 1000)[1:], dtype=torch.float32)[:, None, None]
sampled_posterior = []

for i in range(10000):
    print("Iteration:", i)
    x_0 = torch.randn((1, 1), dtype=torch.float32)
    sampled_post = diff.euler_maruyama(x_0=x_0, observed_data=torch.tensor(observed_data, dtype=torch.float32)[:, None], times=discretization_times, tau=1, network=perceptron)
    sampled_posterior.append(sampled_post)

sampled_posterior = np.array(sampled_posterior)

low = np.min(sampled_posterior)
high = np.max(sampled_posterior)
xx = np.linspace(low, high, 10000)
y = np.array([compute_prior(x)*compute_likelihood(x, observed_data) for x in xx])
plt.hist(sampled_posterior[:, -1, 0], bins=40, density=True, alpha=0.5)
plt.hist(sorted_params[:10000, 0], bins=40, density=True, alpha=0.5)
plt.plot(xx, y)
plt.show()
