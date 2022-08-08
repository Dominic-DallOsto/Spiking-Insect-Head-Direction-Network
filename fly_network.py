import brian2 as b2
import spiking_network
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

neuron_eqs_e = '''
	dv/dt = 1/tau_e * ((v_rest_e - v) + (I_synE+I_synI)/g_mem_e) + sigma_noise_e * sqrt(2/tau_e)*xi 	: volt (unless refractory)
	I_synE = ge * nS * (E_e_exc-v)																		: amp
	I_synI = gi * nS * (E_e_inh-v)																		: amp
	dge/dt = -ge/(tau_syn_e_exc)																		: 1
	dgi/dt = -gi/(tau_syn_e_inh)																		: 1
'''

neuron_eqs_i = '''
	dv/dt = 1/tau_i * ((v_rest_i - v) + (I_synE+I_synI)/g_mem_i) + sigma_noise_i * sqrt(2/tau_i)*xi		: volt (unless refractory)
	I_synE = ge * nS * (E_i_exc-v)																		: amp
	I_synI = gi * nS * (E_i_inh-v)																		: amp
	dge/dt = -ge/(tau_syn_i_exc)																		: 1
	dgi/dt = -gi/(tau_syn_i_inh)																		: 1
'''

default_params = {
	'v_rest_e': -65. * b2.mV,
	'v_rest_i': -60. * b2.mV,
	'v_reset_e': -65. * b2.mV,
	'v_reset_i': -45. * b2.mV,
	'v_thresh_e': -52. * b2.mV,
	'v_thresh_i': -40. * b2.mV,
	'refrac_e': 5. * b2.ms,
	'refrac_i': 2. * b2.ms,
	'tau_e': 100. * b2.ms,
	'tau_i': 10. * b2.ms,
	'tau_syn_e_exc': 10. * b2.ms,
	'tau_syn_e_inh': 2. * b2.ms,
	'tau_syn_i_exc': 10. * b2.ms,
	'tau_syn_i_inh': 2. * b2.ms,
	'E_e_exc': 0. * b2.mV,
	'E_e_inh': -100. * b2.mV,
	'E_i_exc': 0. * b2.mV,
	'E_i_inh': -85. * b2.mV,
	'g_mem_e': 1. * b2.nS,
	'g_mem_i': 1. * b2.nS,
	'weight_ei': 2.,
	'weight_ie': 50.,
	'weight_ii': 2.,
	'weight_input': 20.,
	'sigma_noise_e': 0. * b2.mV,
	'sigma_noise_i': 0. * b2.mV,
}

neuron_args_e = {
	'threshold':'v>v_thresh_e',
	'refractory':'refrac_e',
	'reset':'v=v_reset_e'
}

neuron_args_i = {
	'threshold':'v>v_thresh_i',
	'refractory':'refrac_i',
	'reset':'v=v_reset_i'
}

connectivity_matrix = scipy.io.loadmat('Insect Head Direction Network/connectivity_matrix_drosophila_mine_case_5_9cols_labels1.mat')['con_matrix'].T

# plt.imshow(connectivity_matrix)
# plt.show()

net = spiking_network.SpikingNetwork([16,18,18,8], [neuron_eqs_e, neuron_eqs_e, neuron_eqs_e, neuron_eqs_i], [-65.*b2.mV,-65.*b2.mV,-65.*b2.mV,-60.*b2.mV], [neuron_args_e, neuron_args_e, neuron_args_e, neuron_args_i], default_params)
net.connect_with_connectivity_matrix(connectivity_matrix, ['ge_post += w','ge_post += w','ge_post += w','gi_post += w'])
net.add_poisson_input(2, [100*b2.Hz if 1 <= i <= 3 else 0*b2.Hz for i in range(18)], 40, 'ge_post += w')

out = net.run(500*b2.ms)
net.plot()