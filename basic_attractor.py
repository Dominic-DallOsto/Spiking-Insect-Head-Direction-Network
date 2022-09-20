import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2
import cma
from typing import Callable, Any, List, Tuple

import attractor_network
import parameter_tuner
import analysis

def run_network(network: attractor_network.AttractorNetwork, params: dict):
	network_params = {name : value for name,value in params.items() if not name.startswith('dist')}
	ee_connectivity_profile = np.array([value for name,value in params.items() if name.startswith('dist')])
	
	network.reset()
	network.params.update(network_params)
	network.circular_distance_EE_connectivity(ee_connectivity_profile)
	network.run_with_inputs(warm_up_duration, [100*b2.Hz if 3<=i<=5 else 10*b2.Hz for i in range(network.num_excitatory)])
	network.run_with_inputs(settle_duration+run_duration, 0)

def get_network_results(network: attractor_network.AttractorNetwork, loss: Callable[[np.ndarray,np.ndarray,np.ndarray],Tuple[np.ndarray,float]]):
	start_time = warm_up_duration + settle_duration
	middle_time = start_time + run_duration/2
	response_start = network.monitor_e.i[:][np.logical_and(start_time <= network.monitor_e.t, network.monitor_e.t < middle_time)]
	response_end = network.monitor_e.i[:][network.monitor_e.t >= middle_time]
	indices_start, counts_start = np.unique(response_start, return_counts=True)
	indices_end, counts_end = np.unique(response_end, return_counts=True)
	full_indices_start, full_rates_start = hist_counts_to_full(indices_start, counts_start/(run_duration/2), np.arange(network.num_excitatory))
	full_indices_end, full_rates_end = hist_counts_to_full(indices_end, counts_end/(run_duration/2), np.arange(network.num_excitatory))
	best_fit, rms = loss(full_indices_start, full_rates_start, full_rates_end)
	return full_indices_start, full_rates_start, full_rates_end, best_fit, rms

def hist_counts_to_full(indices, counts, full_range):
	full_counts = np.zeros_like(full_range)
	full_counts[indices] = counts
	return full_range, full_counts

def gaussian(x, mean, stdev, scaled=True):
	return np.exp(-1/2* ((x-mean)/stdev)**2) * (1/(stdev*np.sqrt(2*np.pi)) if scaled else 1)

def circ_wrapped_distance(x:np.ndarray, wrapping_boundary:float=2*np.pi):
	return wrapping_boundary/2 - np.abs(np.abs(x) - wrapping_boundary/2)

def circ_gaussian(x, mean, stdev, scaled=True) -> np.ndarray:
	wrapped_distance = circ_wrapped_distance(x-mean, x.shape[0])
	return np.exp(-1/2* (wrapped_distance/stdev)**2) * (1/(stdev*np.sqrt(2*np.pi)) if scaled else 1)

def circ_triangle(x:np.ndarray, mean):
	negative_wrapped_distance = x.shape[0]/2 - circ_wrapped_distance(x-mean, x.shape[0])
	return negative_wrapped_distance / (x.shape[0]/2)

def circ_sin(x:np.ndarray, mean, width=270):
	distance = 2*np.pi/x.shape[0] * (360/width) * circ_wrapped_distance(x-mean, x.shape[0])
	sin = 1/2*(1 + np.cos(distance))
	sin[abs(distance) > np.pi] = 0
	return sin

def response_rms_error(indices, rates, target):
	full_rates = np.zeros_like(target)
	full_rates[indices] = rates
	return np.sqrt(np.mean((full_rates - target)**2))

def get_rms_and_best(target_rates, rates1, rates2):
	rms = np.sqrt((np.mean((target_rates - np.atleast_2d(rates1).T)**2, axis=0)) + np.mean((target_rates - np.atleast_2d(rates2).T)**2, axis=0)/2)
	best_index = np.argmin(rms)
	return target_rates[:,best_index], rms[best_index]

def get_best_gaussian_rms(indices, rates1, rates2, peak=100, stdev=1): # don't go higher than 100 for this
	mean_range = np.atleast_2d(np.arange(0, len(indices), 0.01))
	gaussian_rates = peak * circ_gaussian(np.tile(np.atleast_2d(indices).T, (1,len(mean_range))), mean_range, stdev, False)
	return get_rms_and_best(gaussian_rates, rates1, rates2)

def get_best_sinusoid_rms(indices, rates1, rates2, peak=100, width=270):
	mean_range = np.atleast_2d(np.arange(0, len(indices), 0.01))
	sinusoid_rates = peak * circ_sin(np.tile(np.atleast_2d(indices).T, (1,len(mean_range))), mean_range, width)
	return get_rms_and_best(sinusoid_rates, rates1, rates2)

def get_best_triangle_rms(indices, rates1, rates2, peak=100):
	mean_range = np.atleast_2d(np.arange(0, len(indices), 0.01))
	triangle_rates = peak * circ_triangle(np.tile(np.atleast_2d(indices).T, (1,len(mean_range))), mean_range)
	return get_rms_and_best(triangle_rates, rates1, rates2)

def windowed_spike_rates(times, indices, duration:int, max_index: int, window_size:int):
	# split the indices array based on the times, with bins depending on window_size
	indices_bins = analysis.split_spike_indices_by_window(times, indices, window_size, duration)
	means_peaks_angles =np.array([bump_population_vector_readout(hist_counts_to_full(*np.unique(indices_array, return_counts=True), np.arange(network.num_excitatory))[1]) for indices_array in indices_bins])
	return means_peaks_angles[:,0], means_peaks_angles[:,1], means_peaks_angles[:,2]

def bump_population_vector_readout(rates):
	"""Takes the firing rates for each neuron, distributed around a circle and calculates the mean angle and standard deviation."""
	angles = np.linspace(0, 2*np.pi, len(rates)+1)[:-1]
	mean_x = np.mean(rates*np.cos(angles))
	mean_y = np.mean(rates*np.sin(angles))
	mean_angle = np.arctan2(mean_y, mean_x)
	peak = np.sqrt(mean_x**2 + mean_y**2)
	# treat each spike as a sample
	standard_deviation = np.sqrt(np.sum(rates*circ_wrapped_distance(angles-mean_angle)**2) / np.sum(rates))
	return mean_angle, peak, standard_deviation

def plot_response(network, loss):
	indices, rates_start, rates_end, best_fit, rms = get_network_results(network, loss)
	# pop_mean, pop_peak, pop_stddev = bump_population_vector_readout(rates)
	window_duration=50
	means, _, angles = windowed_spike_rates(network.monitor_e.t/b2.ms, network.monitor_e.i[:], int(total_duration/b2.ms), network.num_inhibitory, window_duration)
	neuron_means = (means % (2*np.pi))/2/np.pi*network.num_excitatory
	neuron_angles = angles/2/np.pi*network.num_excitatory

	fig, axs = plt.subplots(3, 1, figsize=(12,4), sharex=True)
	assert(isinstance(axs, np.ndarray))
	axs[0].plot(network.monitor_e.t/b2.ms, network.monitor_e.i[:], '.b')
	popvec_time = window_duration/2+np.arange(0,int(total_duration/b2.ms),window_duration)
	axs[0].fill_between(popvec_time, neuron_means-neuron_angles, neuron_means+neuron_angles, alpha=0.2, color='lime')
	axs[0].set_ylabel('Neuron index')
	axs[0].set_ylim([0,network.num_excitatory])
	axs[0].set_title('E spikes')
	axs[1].plot(network.monitor_i.t/b2.ms, network.monitor_i.i[:], '.r')
	axs[1].set_ylabel('Neuron index')
	axs[1].set_title('I spikes')
	axs[2].plot(network.monitor_input.t/b2.ms, network.monitor_input.i[:], '.k')
	axs[2].set_xlabel('Time (ms)')
	axs[2].set_ylabel('Neuron index')
	axs[2].set_title('Input spikes')
	plt.tight_layout()

	fig, ax = plt.subplots(1,1)
	ax.set_title('Self-sustained spike rates')
	ax.set_xlabel('Neuron index')
	ax.set_ylabel('Firing Rate (Hz)')
	# ax.set_xlim([0,network.num_excitatory-1])
	ax.bar(indices, rates_start, align='edge', width=-0.3, label='actual firing first half')
	ax.bar(indices, rates_end, align='edge', width=0.3, label='actual firing second half')
	ax.plot(np.arange(network.num_excitatory), best_fit, 'k.-', label='target')
	plt.legend()
	# ax.plot(pop_peak*circ_gaussian(indices, pop_mean*network.num_excitatory/2/np.pi, pop_stddev*network.num_excitatory/2/np.pi, False), 'r')
	ax.set_title(f'Self-sustained spike rates\nerror={rms:.3f}')
	plt.show()

def plot_results(network: attractor_network.AttractorNetwork, axs: list[plt.Axes], loss: Callable[[np.ndarray,np.ndarray,np.ndarray],Tuple[np.ndarray,float]]):
	for ax in axs:
		for plot in ax.lines + ax.collections + ax.containers:
			plot.remove()

	indices, rates_start, rates_end, best_fit, rms = get_network_results(network, loss)
	# pop_mean, pop_peak, pop_stddev = bump_population_vector_readout(rates)
	window_duration=100
	means, _, angles = windowed_spike_rates(network.monitor_e.t/b2.ms, network.monitor_e.i[:], int(total_duration/b2.ms), network.num_inhibitory, window_duration)
	neuron_means = (means % (2*np.pi))/2/np.pi*network.num_excitatory
	neuron_angles = angles/2/np.pi*network.num_excitatory

	axs[0].plot(network.monitor_e.t/b2.ms, network.monitor_e.i[:], '.b')
	popvec_time = window_duration/2+np.arange(0,int(total_duration/b2.ms),window_duration)
	axs[0].fill_between(popvec_time, neuron_means-neuron_angles, neuron_means+neuron_angles, alpha=0.2, color='lime')
	axs[0].set_ylabel('Neuron index')
	axs[0].set_ylim([0,network.num_excitatory])
	axs[0].set_title('E spikes')
	axs[1].plot(network.monitor_i.t/b2.ms, network.monitor_i.i[:], '.r')
	axs[1].set_ylabel('Neuron index')
	synchrony = analysis.spike_train_synchrony_correlation(network.monitor_i.t/b2.ms, network.monitor_i.i[:], int(total_duration/b2.ms))
	axs[1].set_title(f'I spikes - synchrony = {synchrony}')
	axs[2].plot(network.monitor_input.t/b2.ms, network.monitor_input.i[:], '.k')
	axs[2].set_xlabel('Time (ms)')
	axs[2].set_ylabel('Neuron index')
	axs[2].set_title('Input spikes')
	plt.tight_layout()

	axs[3].set_title('Self-sustained spike rates')
	axs[3].set_xlabel('Neuron index')
	axs[3].set_ylabel('Firing Rate (Hz)')
	axs[3].bar(indices, rates_start, align='edge', width=-0.3, label='actual firing first half')
	axs[3].bar(indices, rates_end, align='edge', width=0.3, label='actual firing second half')
	axs[3].plot(np.arange(network.num_excitatory), best_fit, 'k.-', label='target')
	axs[3].legend()
	# axs[3].plot(pop_peak*circ_gaussian(indices, pop_mean*network.num_excitatory/2/np.pi, pop_stddev*network.num_excitatory/2/np.pi, False), 'r')
	axs[3].set_title(f'Self-sustained spike rates\nerror={rms:.3f}')

	axs[0].get_figure().canvas.draw_idle()
	axs[3].get_figure().canvas.draw_idle()

def setup_plot_wrapper(network: attractor_network.AttractorNetwork):
	def setup_plot():
		fig, axs = plt.subplots(3, 1, figsize=(12,4), sharex=True)
		assert(isinstance(axs, np.ndarray))
		axs[0].plot(network.monitor_e.t/b2.ms, network.monitor_e.i[:], '.b')
		axs[0].set_ylabel('Neuron index')
		axs[0].set_ylim([0,network.num_excitatory])
		axs[0].set_title('E spikes')
		axs[1].plot(network.monitor_i.t/b2.ms, network.monitor_i.i[:], '.r')
		axs[1].set_ylabel('Neuron index')
		axs[1].set_title('I spikes')
		axs[2].plot(network.monitor_input.t/b2.ms, network.monitor_input.i[:], '.k')
		axs[2].set_xlabel('Time (ms)')
		axs[2].set_ylabel('Neuron index')
		axs[2].set_title('Input spikes')
		plt.tight_layout()

		fig, ax = plt.subplots(1,1)
		ax.set_title('Self-sustained spike rates')
		ax.set_xlabel('Neuron index')
		ax.set_ylabel('Firing Rate (Hz)')
		ax.set_xlim([-0.5,network.num_excitatory-0.5])

		return np.hstack((axs, np.array(ax)))
	return setup_plot


network = attractor_network.AttractorNetwork()
warm_up_duration = 200*b2.ms
settle_duration = 200*b2.ms
run_duration = 400*b2.ms
total_duration = warm_up_duration + settle_duration + run_duration

params_to_tune = {
	'dist0': (0,20,10, 1),
	'dist1': (0,20,5, 1),
	'dist2': (0,20,2, 1),
	'dist3': (0,20,1, 1),
	'dist4': (0,20,0, 1),
	'weight_ei': (0, 50, 15, 1),
	'weight_ie': (0, 500, 85, 1),
	'weight_ii': (0, 10, 5, 1),
	'sigma_noise_e': (0, 50, 20, b2.mV),
}

def run_get_loss(params, loss):
	params_dict = {name : value*unit for (name, (_,_,_,unit)),value in zip(params_to_tune.items(), params)}
	run_network(network, params_dict)
	_, _, _, _, error = get_network_results(network, loss)

	return error

x0 = [p[2] for p in params_to_tune.values()]
bounds = [p[:2] for p in params_to_tune.values()]

# return function that takes a value between 0 and 1 and outputs a value in the range
param_scaler = lambda min,max : lambda x: min+x*(max-min)
param_scaler_nonnegative = lambda min,max : lambda x: min+x**2*(max-min)
param_unscaler_nonnegative = lambda min,max : lambda x: np.sqrt((x-min)/(max-min))

param_scalers = [param_scaler_nonnegative(*bound) for bound in bounds]
param_unscalers = [param_unscaler_nonnegative(*bound) for bound in bounds]

scale_params = lambda x: [scaler(param) for scaler,param in zip(param_scalers, x)]
unscale_params = lambda x: [scaler(param) for scaler,param in zip(param_unscalers, x)]

# return a function that wraps run_get_loss, so that CMA can learn params in the range 0-1 but these are 
# transformed to the ranges we want for the simulation
# returns a function that takes x, calls run_get_loss(transform_params(x))
def simulation_param_wrapper(loss):
	return lambda x: run_get_loss(scale_params(x), loss)

def run_sim(axs, sliders):
	def run(_):
		params = {slider.label._text : slider.val*unit for slider, (_,_,_,unit) in zip(sliders, params_to_tune.values())}
		run_network(network, params)
		plot_results(network, axs, get_best_gaussian_rms)
	return run

params_to_tune_no_units = {name: value[:3] for name, value in params_to_tune.items()}
parameter_tuner = parameter_tuner.ParameterTuner(params_to_tune_no_units, setup_plot_wrapper(network), run_sim)
parameter_tuner.start()

# scipy.optimize.minimize(run_get_loss, x0, method='BFGS', options={'disp':True, 'eps':1})

bounds = [p[:2] for p in params_to_tune.values()]

# scipy.optimize.differential_evolution(run_get_loss, bounds, disp=True)
''' nice params - took like 10000 iterations though
4.063 [ 4.98113731  4.47966305  2.34875325 85.43293882 97.80732447  6.66681116]
3.931 [ 6.66436962  4.00808527  3.25404152 92.76669626 68.78163303  7.04950664]
3.615 [ 6.43519855  3.56683217  2.26724691 70.64832973 36.28805101  5.72906753]
2.425 [ 4.90522101  4.50705598  2.58773827 78.15313821 80.72587409  7.51129724]
'''

# return function that takes a value between 0 and 1 and outputs a value in the range
param_scaler = lambda min,max : lambda x: min+x*(max-min)
param_scaler_nonnegative = lambda min,max : lambda x: min+x**2*(max-min)

# return a function that wraps run_get_loss, so that CMA can learn params in the range 0-1 but these are 
# transformed to the ranges we want for the simulation
# returns a function that takes x, calls run_get_loss(transform_params(x))
def param_wrapper():
	param_wrappers = [param_scaler_nonnegative(*bound) for bound in bounds]
	return lambda x: run_get_loss([scaler(param) for scaler,param in zip(param_wrappers, x)], get_best_gaussian_rms)

x0_unscaled = [np.sqrt((p[2]-p[0])/(p[1]-p[0])) for p in params_to_tune.values()]

# xopt, es = cma.fmin2(param_wrapper(), x0_unscaled, 0.1, options={'maxfevals': 2e2})
# cma.plot()
# cma.s.figshow()
# cma.s.figsave('fig.png')
