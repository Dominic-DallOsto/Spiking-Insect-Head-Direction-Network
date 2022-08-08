import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

## N populations of neurons - how big
# equations for each neuron population
# how to do synapses?

class SpikingNetwork():
	'''
	A Brian2 spiking network with different populations of interconnected neurons.
	'''
	def __init__(self, population_sizes:list[int], population_equations:list[str], initial_voltages:list[b2.Quantity], population_arguments:list[dict], population_parameters:dict) -> None:
		assert len(population_sizes) == len(population_equations), "SpikingNetwork: population_sizes and population_equations must be same length"
		assert len(population_sizes) == len(population_equations), "SpikingNetwork: population_sizes and population_parameters must be same length"

		self.parameters = population_parameters
		self.population_size_ranges = []

		self.network = b2.Network()
		self.populations = []
		self.spike_monitors = []
		for pop_size, equations, initial_voltage, args in zip(population_sizes, population_equations, initial_voltages, population_arguments):
			neuron_group = b2.NeuronGroup(pop_size, equations, **args, method='euler')
			neuron_group.v = initial_voltage
			self.populations.append(neuron_group)

			spike_monitor = b2.SpikeMonitor(neuron_group)
			self.spike_monitors.append(spike_monitor)

			last_neuron_index = self.population_size_ranges[-1][1] if len(self.population_size_ranges) > 0 else 0
			self.population_size_ranges.append((last_neuron_index, last_neuron_index+pop_size))
		self.network.add(self.populations + self.spike_monitors)

	def connect_with_connectivity_matrix(self, connectivity_matrix:np.ndarray, synapse_equations:list[str]):
		self.connections = []
		for presynaptic_population_range, presynaptic_population, equation in zip(self.population_size_ranges, self.populations, synapse_equations):
			for postsynaptic_population_range, postsynaptic_population in zip(self.population_size_ranges, self.populations):
				connectivity = connectivity_matrix[presynaptic_population_range[0]:presynaptic_population_range[1],
				                                   postsynaptic_population_range[0]:postsynaptic_population_range[1]]
				connection = b2.Synapses(presynaptic_population, postsynaptic_population, 'w : 1', on_pre=equation)
				connection.connect(True)
				connection.w = connectivity[connection.i, connection.j]
				self.connections.append(connection)
		self.network.add(self.connections)

	def add_poisson_input(self, population_index:int, rates:list[b2.Quantity|str]|b2.Quantity|str, weight:float|str, equation:str):
		input_group = b2.PoissonGroup(self.populations[population_index].N, rates)
		input_connection = b2.Synapses(input_group, self.populations[population_index], f'w = {weight} : 1', on_pre=equation)
		input_connection.connect('i == j')
		self.network.add([input_group,input_connection])

	def run(self, duration):
		self.network.run(duration, namespace=self.parameters)
		return self.spike_monitors

	def plot(self, timescale=b2.ms):
		fig, axs = plt.subplots(len(self.populations), 1, sharex=True)
		assert(isinstance(axs, np.ndarray))

		for i, spike_monitor in enumerate(self.spike_monitors):
			axs[i].plot(spike_monitor.t/timescale, spike_monitor.i[:], '.')

		plt.xlabel(f'time ({timescale})')
		plt.tight_layout()
		plt.show()


# Diehl and Cook neuron model from https://github.com/sdpenguin/Brian2STDPMNIST/blob/2d935d0a98b6c94cfbc1cb8304f16a578a57342b/Diehl%26Cook_spiking_MNIST_Brian2.py#L204
# Rearranged and parameterised these equations for flexibility
# Added stochastic noise to the voltage update equations - https://brian2.readthedocs.io/en/stable/user/models.html#noise
# neuron_eqs_e = '''
# 	dv/dt = 1/tau_e * ((v_rest_e - v) + (I_synE+I_synI)/g_mem_e) + sigma_noise_e * sqrt(2/tau_e)*xi 	: volt (unless refractory)
# 	I_synE = ge * nS * (E_e_exc-v)																		: amp
# 	I_synI = gi * nS * (E_e_inh-v)																		: amp
# 	dge/dt = -ge/(tau_syn_e_exc)																		: 1
# 	dgi/dt = -gi/(tau_syn_e_inh)																		: 1
# '''

# neuron_eqs_i = '''
# 	dv/dt = 1/tau_i * ((v_rest_i - v) + (I_synE+I_synI)/g_mem_i) + sigma_noise_i * sqrt(2/tau_i)*xi		: volt (unless refractory)
# 	I_synE = ge * nS * (E_i_exc-v)																		: amp
# 	I_synI = gi * nS * (E_i_inh-v)																		: amp
# 	dge/dt = -ge/(tau_syn_i_exc)																		: 1
# 	dgi/dt = -gi/(tau_syn_i_inh)																		: 1
# '''

# default_params = {
# 	'v_rest_e': -65. * b2.mV,
# 	'v_rest_i': -60. * b2.mV,
# 	'v_reset_e': -65. * b2.mV,
# 	'v_reset_i': -45. * b2.mV,
# 	'v_thresh_e': -52. * b2.mV,
# 	'v_thresh_i': -40. * b2.mV,
# 	'refrac_e': 5. * b2.ms,
# 	'refrac_i': 2. * b2.ms,
# 	'tau_e': 100. * b2.ms,
# 	'tau_i': 10. * b2.ms,
# 	'tau_syn_e_exc': 10. * b2.ms,
# 	'tau_syn_e_inh': 2. * b2.ms,
# 	'tau_syn_i_exc': 10. * b2.ms,
# 	'tau_syn_i_inh': 2. * b2.ms,
# 	'E_e_exc': 0. * b2.mV,
# 	'E_e_inh': -100. * b2.mV,
# 	'E_i_exc': 0. * b2.mV,
# 	'E_i_inh': -85. * b2.mV,
# 	'g_mem_e': 1. * b2.nS,
# 	'g_mem_i': 1. * b2.nS,
# 	'weight_ei': 2.,
# 	'weight_ie': 50.,
# 	'weight_ii': 2.,
# 	'weight_input': 20.,
# 	'sigma_noise_e': 5. * b2.mV,
# 	'sigma_noise_i': 5. * b2.mV,
# }


# class AttractorNetwork():
# 	'''
# 	Attractor network with E and I neurons with synaptic current dynamics
# 	'''
# 	def __init__(self, num_excitatory:int=8, num_inhibitory:int=8) -> None:
# 		self.num_excitatory = num_excitatory
# 		self.num_inhibitory = num_inhibitory

# 		self.default_params = default_params.copy()
# 		self.params = self.default_params.copy()

# 		self.network = b2.Network()
# 		neuron_group_e = b2.NeuronGroup(self.num_excitatory, neuron_eqs_e, threshold='v>v_thresh_e', refractory='refrac_e', reset='v=v_reset_e', method='euler')
# 		neuron_group_i = b2.NeuronGroup(self.num_inhibitory, neuron_eqs_i, threshold='v>v_thresh_i', refractory='refrac_i', reset='v=v_reset_i', method='euler')
# 		neuron_group_e.v = self.params['v_rest_e']
# 		neuron_group_i.v = self.params['v_rest_i']

# 		self.ee_connection = ee_connection = b2.Synapses(neuron_group_e, neuron_group_e, 'w : 1', on_pre='ge_post += w')
# 		ee_connection.connect(True)

# 		ei_connection = b2.Synapses(neuron_group_e, neuron_group_i, 'w = weight_ei : 1', on_pre='ge_post += w')
# 		ei_connection.connect(True)

# 		ie_connection = b2.Synapses(neuron_group_i, neuron_group_e, 'w = weight_ie : 1', on_pre='gi_post += w')
# 		ie_connection.connect('i == j')

# 		ii_connection = b2.Synapses(neuron_group_i, neuron_group_i, 'w = weight_ii : 1', on_pre='gi_post += w')
# 		ii_connection.connect(True)

# 		self.neuron_group_input = b2.PoissonGroup(self.num_excitatory, 0*b2.Hz)
# 		input_connection = b2.Synapses(self.neuron_group_input, neuron_group_e, 'w = weight_input : 1', on_pre='ge_post += w')
# 		input_connection.connect('i == j')

# 		self.monitor_e = monitor_e = b2.SpikeMonitor(neuron_group_e)
# 		self.monitor_i = monitor_i = b2.SpikeMonitor(neuron_group_i)
# 		self.monitor_input = monitor_input = b2.SpikeMonitor(self.neuron_group_input)

# 		self.network.add([neuron_group_e, neuron_group_i, self.neuron_group_input, ee_connection, ei_connection, ie_connection, ii_connection, input_connection, monitor_e, monitor_i, monitor_input])

# 		self.network.store()
	
# 	def reset(self):
# 		self.params = self.default_params.copy()
# 		self.network.restore()

# 	def run_with_inputs(self, duration, input_rates):
# 		self.neuron_group_input.rates = input_rates
# 		self.network.run(duration, namespace=self.params)

# 	def set_EE_connectivity(self, weight_matrix: np.ndarray):
# 		assert weight_matrix.ndim == 2, 'EE weight matrix should be 2D'
# 		assert weight_matrix.shape[0] == self.num_excitatory, f'weight matrix should have {self.num_excitatory} size along first dimension'
# 		assert weight_matrix.shape[1] == self.num_excitatory, f'weight matrix should have {self.num_excitatory} size along second dimension'

# 		self.ee_connection.w = weight_matrix[self.ee_connection.i, self.ee_connection.j]

# 	def circular_distance_EE_connectivity(self, weight_profile: np.ndarray):
# 		half_neuron_number = self.num_excitatory // 2
# 		assert weight_profile.shape[0] == half_neuron_number + 1, f'weight profile should have {half_neuron_number + 1} elements'

# 		circular_distance = half_neuron_number - abs(abs(self.ee_connection.i-self.ee_connection.j) - half_neuron_number)
# 		self.ee_connection.w = weight_profile[circular_distance]

# 	def gaussian_EE_connectivity(self, max: float, stddev: float):
# 		half_neuron_number = self.num_excitatory // 2
# 		circular_distance = half_neuron_number - abs(abs(self.ee_connection.i-self.ee_connection.j) - half_neuron_number)
# 		self.ee_connection.w = max*np.exp(-1/2 * (circular_distance/stddev)**2)


# def merge_dictionaries(dictionaries):
# 	return { key: [dictionary.get(key) for dictionary in dictionaries if key in dictionary] for key in set().union(*dictionaries) }