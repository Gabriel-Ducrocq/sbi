import numpy as np
from bisect import bisect

def get_dataset(distributions, datasets):
    """
    Get the distribution number based on sampled time: if sampled time is t_n < t < t_n-1, then we should keep the samples
     of distrib n an n-1
    :param distributions: dictionnary containing informations on the distributions
    :param datasets: dictionnary containing sampled data from the distributions
    :return: a dictionnary with the starting and ending point for each sampled time
    """
    n_times = len(distributions)
    times = []
    all_datasets = []
    for i in range(n_times):
        distrib_name = "distrib" + str(i)
        times.append(distributions[distrib_name]["t"])
        all_datasets.append(datasets[distrib_name])

    all_datasets = np.concatenate(all_datasets, axis=1)
    belongings = np.array(list((map(lambda t: bisect(times, t), datasets["sampled_times"]))))
    starting_points = []
    ending_points = []
    sampled_time_difference = []
    time_between_distrib = []
    distrib_number = []
    for i, all_data_t in enumerate(all_datasets):
        starting_points.append(all_data_t[belongings[i]-1])
        print(i)
        ending_points.append(all_data_t[belongings[i]])
        sampled_time_difference.append(datasets["sampled_times"][i] - times[belongings[i-1]])
        distrib_number.append(belongings[i])
        time_between_distrib.append(times[belongings[i] - belongings[i-1]])


    final_dataset = {"starting_point": np.array(starting_points), "ending_point":np.array(ending_points),
                     "sampled_time_difference":np.array(sampled_time_difference),
                     "distribution_number":np.array(distrib_number), "time_between_distrib":time_between_distrib
                     }

    return final_dataset

