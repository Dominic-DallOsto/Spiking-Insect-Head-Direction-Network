import brian2 as b2
import numpy as np

# Diehl and Cook neuron model from https://github.com/sdpenguin/Brian2STDPMNIST/blob/2d935d0a98b6c94cfbc1cb8304f16a578a57342b/Diehl%26Cook_spiking_MNIST_Brian2.py#L204
# Added stochastic noise to the voltage update equations
neuron_eqs_e = '''
	dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms) + v_noise_e*sqrt(2/100*ms)*xi : volt (unless refractory)
	I_synE = ge * nS *         -v                           : amp
	I_synI = gi * nS * (-100.*mV-v)                          : amp
	dge/dt = -ge/(tau_e)                                   : 1
	dgi/dt = -gi/(2.0*ms)                                  : 1
	theta      :volt
'''

neuron_eqs_i = '''
	dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms) + v_noise_i*sqrt(2/10*ms)*xi : volt (unless refractory)
	I_synE = ge * nS *         -v                           : amp
	I_synI = gi * nS * (-85.*mV-v)                          : amp
	dge/dt = -ge/(1.0*ms)                                   : 1
	dgi/dt = -gi/(2.0*ms)                                  : 1
'''

synapse_eqs_e_gaussian = '''
w = weight_ee*exp(-1/2 * (closest_dist/stdev_ee)**2) : 1
closest_dist = N_pre/2 - abs(abs(i_pre-i_post) - N_pre/2) : 1
'''

default_params = {
	'v_rest_e': -65. * b2.mV,
	'v_rest_i': -60. * b2.mV,
	'v_reset_e': -65. * b2.mV,
	'v_reset_i': -45. * b2.mV,
	'v_thresh_e': -52. * b2.mV,
	'v_thresh_i': -40. * b2.mV,
	'offset': 20.0*b2.mV,
	'refrac_e': 5. * b2.ms,
	'refrac_i': 2. * b2.ms,
	'tau_e': 5.0*b2.ms,
	'weight_ee': 15,
	'stdev_ee': 1.5,
	'weight_ei': 2,
	'weight_ie': 50,
	'weight_ii': 2,
	'weight_input': 20,
	'v_noise_e': 20*b2.mV/b2.ms,
	'v_noise_i': 10*b2.mV/b2.ms,
}


class AttractorNetwork():
	'''
	Attractor network with E and I neurons with synaptic current dynamics
	'''
	def __init__(self, num_excitatory:int=8, num_inhibitory:int=8, weight_matrix_connectivity=True) -> None:
		self.num_excitatory = num_excitatory
		self.num_inhibitory = num_inhibitory

		v_thresh_e_str = 'v>v_thresh_e'
		v_thresh_i_str = 'v>v_thresh_i'
		v_reset_i_str = 'v=v_reset_i'
		scr_e = 'v = v_reset_e; timer = 0*ms'

		self.default_params = default_params.copy()
		self.params = self.default_params.copy()

		self.network = b2.Network()
		neuron_group_e = b2.NeuronGroup(self.num_excitatory, neuron_eqs_e, threshold=v_thresh_e_str, refractory='refrac_e', reset=scr_e, method='euler')
		neuron_group_i = b2.NeuronGroup(self.num_inhibitory, neuron_eqs_i, threshold=v_thresh_i_str, refractory='refrac_i', reset=v_reset_i_str, method='euler')
		neuron_group_e.v = self.params['v_rest_e']
		neuron_group_i.v = self.params['v_rest_i']

		if weight_matrix_connectivity:
			self.ee_connection = ee_connection = b2.Synapses(neuron_group_e, neuron_group_e, 'w : 1', on_pre='ge_post += w')
		else:
			self.ee_connection = ee_connection = b2.Synapses(neuron_group_e, neuron_group_e, synapse_eqs_e_gaussian, on_pre='ge_post += w')
		ee_connection.connect(True)

		ei_connection = b2.Synapses(neuron_group_e, neuron_group_i, 'w = weight_ei : 1', on_pre='ge_post += w')
		ei_connection.connect(True)

		ie_connection = b2.Synapses(neuron_group_i, neuron_group_e, 'w = weight_ie : 1', on_pre='gi_post += w')
		ie_connection.connect('i == j')

		ii_connection = b2.Synapses(neuron_group_i, neuron_group_i, 'w = weight_ii : 1', on_pre='gi_post += w')
		ii_connection.connect(True)

		self.neuron_group_input = b2.PoissonGroup(self.num_excitatory, 0*b2.Hz)
		input_connection = b2.Synapses(self.neuron_group_input, neuron_group_e, 'w = weight_input : 1', on_pre='ge_post += w')
		input_connection.connect('i == j')

		self.monitor_e = monitor_e = b2.SpikeMonitor(neuron_group_e)
		self.monitor_i = monitor_i = b2.SpikeMonitor(neuron_group_i)
		self.monitor_input = monitor_input = b2.SpikeMonitor(self.neuron_group_input)

		self.network.add([neuron_group_e, neuron_group_i, self.neuron_group_input, ee_connection, ei_connection, ie_connection, ii_connection, input_connection, monitor_e, monitor_i, monitor_input])

		self.network.store()
	
	def reset(self):
		self.params = self.default_params.copy()
		self.network.restore()

	def run_with_inputs(self, duration, input_rates):
		self.neuron_group_input.rates = input_rates
		self.network.run(duration, namespace=self.params)

	def set_EE_connectivity(self, weight_matrix: np.ndarray):
		assert weight_matrix.ndim == 2, 'EE weight matrix should be 2D'
		assert weight_matrix.shape[0] == self.num_excitatory, f'weight matrix should have {self.num_excitatory} size along first dimension'
		assert weight_matrix.shape[1] == self.num_excitatory, f'weight matrix should have {self.num_excitatory} size along second dimension'

		self.ee_connection.w = weight_matrix[self.ee_connection.i, self.ee_connection.j]

	def circular_distance_EE_connectivity(self, weight_profile: np.ndarray):
		half_neuron_number = self.num_excitatory // 2
		assert weight_profile.shape[0] == half_neuron_number + 1, f'weight profile should have {half_neuron_number + 1} elements'

		circular_distance = half_neuron_number - abs(abs(self.ee_connection.i-self.ee_connection.j) - half_neuron_number)
		self.ee_connection.w = weight_profile[circular_distance]