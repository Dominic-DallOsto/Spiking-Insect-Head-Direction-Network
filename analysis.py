import numpy as np


def split_spike_indices_by_window(times, indices, window_size:int, total_duration:int):
	"""
	Split a list of spike indices (with accompanying spike times) into bins 
	based on equal length time windows.
	"""
	return np.split(indices, np.sum(np.atleast_2d(times) < np.atleast_2d(np.arange(window_size,total_duration,window_size)).T, axis=1))


# given a spike train (times and indices of spikes)
# - split the spike train into small time windows
# - for each time window calculate the number of spikes for each neuron
# - for each pair of neurons, calculate the correlation between the time windowed spike counts
# - return the average correlation as a measure of population spiking synchrony
# 
def spike_train_synchrony_correlation(spike_times, spike_indices, total_duration:int):
	"""
	Calculate the average correlation between the windowed spike counts for all pairs
	of neurons in the network.
	Returns NaN if no neurons fired.

	From: 
	Kumar, A., Schrader, S., Aertsen, A., & Rotter, S. (2008). 
	The High-Conductance State of Cortical Networks. Neural Computation, 20(1), 1-43. 
	https://doi.org/10.1162/neco.2008.20.1.1

	"""
	neuron_indices = np.unique(spike_indices)
	num_neurons = len(neuron_indices)
	windowed_spikes = split_spike_indices_by_window(spike_times, spike_indices, 2, total_duration)
	windowed_spike_counts = np.zeros((1+np.max(neuron_indices), len(windowed_spikes)))

	for i, window in enumerate(windowed_spikes):
		spike_indices, spike_counts = np.unique(window, return_counts=True)
		windowed_spike_counts[spike_indices, i] = spike_counts
	
	# only calculate the correlations for spike trains where we have spikes
	correlation_coefficients = np.tril(np.corrcoef(windowed_spike_counts[neuron_indices,:]), k=-1)
	num_coefficients = num_neurons*(num_neurons-1) / 2
	# on the off chance that a spike train is perfectly regular, correlating with it will
	# result in a NaN value - remove these
	return np.nansum(correlation_coefficients) / (num_coefficients - np.count_nonzero(np.isnan(correlation_coefficients)))