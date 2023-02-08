import numpy as np
import dataset

class Sampler:
    def __init__(self, N_samples):
        self.N_samples = N_samples

    def sample_prior(self, prior):
        """
        Samples parameters from the prior distribution
        :param prior: python function, sampling a parameter on calling
        :return: numpy array (N_samples, parameter_dimension)
        """
        sampled_params = [prior() for _ in range(self.N_samples)]
        return np.array(sampled_params)

    def samples_likelihood(self, likelihood, sampled_params):
        """
        Samples data from the likelihood given the parameters
        :param likelihood: python function, sampling a datapoint given a parameter on calling
        :param sampled_params: numpy array (N_samples, parameter_dimension), parameters sampled from likelihood
        :return: numpy array (N_samples, data_dimension)
        """
        return np.array(list(map(lambda param: likelihood(param), sampled_params)))

    def sample(self, prior, likelihood):
        """
        Forward sampling of a dataset
        :param prior: python function, sampling a parameter on calling
        :param likelihood: python function, sampling a datapoint given a parameter on calling
        :return: a Dataset object
        """
        sampled_parameters = self.sample_prior(prior)
        sampled_data = self.samples_likelihood(likelihood, sampled_parameters)
        sampled_dataset = dataset.Dataset(sampled_parameters, sampled_data)
        return sampled_dataset