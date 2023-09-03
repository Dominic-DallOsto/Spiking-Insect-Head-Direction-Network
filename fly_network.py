import brian2 as b2
import spiking_network
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import parameter_tuner
import connectivity_matrices

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
	'v_rest_e': -52. * b2.mV,
	'v_rest_i': -52. * b2.mV,
	'v_reset_e': -52. * b2.mV,
	'v_reset_i': -52. * b2.mV,
	'v_thresh_e': -45. * b2.mV,
	'v_thresh_i': -45. * b2.mV,
	'refrac_e': 2. * b2.ms,
	'refrac_i': 2. * b2.ms,
	'tau_e': 10.*b2.Mohm*2*b2.nF,
	'tau_i': 10.*b2.Mohm*2*b2.nF,
	'tau_syn_e_exc': 50. * b2.ms,
	'tau_syn_e_inh': 2. * b2.ms,
	'tau_syn_i_exc': 10. * b2.ms,
	'tau_syn_i_inh': 2. * b2.ms,
	'E_e_exc': 0. * b2.mV,
	'E_e_inh': -100. * b2.mV,
	'E_i_exc': 0. * b2.mV,
	'E_i_inh': -85. * b2.mV,
	'g_mem_e': 1./(10*b2.Mohm),
	'g_mem_i': 1./(10*b2.Mohm),
	'sigma_noise_e': 0. * b2.mV,
	'sigma_noise_i': 0. * b2.mV,
	'input_frequency': 1000*b2.Hz,
	'input_noise': 20*b2.Hz,
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

connectivity_matrix_str = connectivity_matrices.get_fly_connectivity_matrix_string()


net = spiking_network.SpikingNetwork([16,18,18,8], [neuron_eqs_e]*3+[neuron_eqs_i], [default_params['v_rest_e']]*3+[default_params['v_rest_i']], [neuron_args_e]*3+[neuron_args_i], default_params)
net.connect_with_named_connectivity_matrix(connectivity_matrix_str, ['ge_post += w','ge_post += w','ge_post += w','gi_post += w'])
net.add_poisson_input(2, '(input_frequency*((i == 4)+(i == 13)) + input_noise)*(t < 500*ms) + (t >= 500*ms)*input_noise', 40, 'ge_post += w')
# net.add_poisson_input(2, [200*b2.Hz if 0 <= i <= 3 else 40*b2.Hz for i in range(18)], 20, 'ge_post += w')
net.network.store()


params_to_tune = { #min, max, default, unit
	'sigma_noise_e': (0, 50, 20, b2.mV),
	'input_frequency': (0, 2000, 1000, b2.Hz),
	'input_noise': (0, 100, 20, b2.Hz),
	'connections_PEN_EPG': (0, 5, 1, 1),
	'connections_PEG_EPG': (0, 5, 1, 1),
	'connections_EPG_PEN': (0, 5, 3, 1),
	'connections_EPG_PEG': (0, 5, 3, 1),
	'connections_EPG_D7': (0, 5, 1.1, 1),
	'connections_D7_PEN': (-5, 0, -1, 1),
	'connections_D7_PEG': (-5, 0, -1, 1),
	'connections_D7_D7': (-5, 0, -1.1, 1),
}

def run_network(network: spiking_network.SpikingNetwork, params: dict):
	network_params = {name : value for name,value in params.items()}
	
	network.network.restore()
	network.parameters.update(network_params)
	network.run(1000*b2.ms)
	return network


def setup_plot_wrapper(network: spiking_network.SpikingNetwork):
	def setup_plot():
		network.plot()

		return plt.gcf().axes
	return setup_plot

def run_sim(axs, sliders):
	def run(_):
		for ax in axs:
			for plot in ax.lines + ax.collections + ax.containers:
				plot.remove()

		params = {slider.label._text : slider.val*unit for slider, (_,_,_,unit) in zip(sliders, params_to_tune.values())}
		run_network(net, params)
		# axs[0].plot(net.spike_monitors[0].t, net.spike_monitors[0].i[:])
		for ax, spike_monitor in zip(axs, net.spike_monitors):
			ax.plot(spike_monitor.t/b2.ms, spike_monitor.i[:], '.')

		axs[0].set_ylabel('P-EN')
		axs[1].set_ylabel('P-EG')
		axs[2].set_ylabel('E-PG')
		axs[3].set_ylabel('Delta7')
		
		axs[0].get_figure().canvas.draw_idle()

		# net.plot()
	return run

params_to_tune_no_units = {name: value[:3] for name, value in params_to_tune.items()}
parameter_tuner = parameter_tuner.ParameterTuner(params_to_tune_no_units, setup_plot_wrapper(net), run_sim)
parameter_tuner.start()

# out = net.run(1000*b2.ms)
# net.plot()
# plt.gcf().axes[0].set_ylabel('P-EN')
# plt.gcf().axes[1].set_ylabel('P-EG')
# plt.gcf().axes[2].set_ylabel('E-PG')
# plt.gcf().axes[3].set_ylabel('Delta7')
# plt.tight_layout()
# net.plot_spike_rates()
# plt.gcf().axes[0].set_title('P-EN')
# plt.gcf().axes[1].set_title('P-EG')
# plt.gcf().axes[2].set_title('E-PG')
# plt.gcf().axes[3].set_title('Delta7')
# plt.tight_layout()
# plt.show()

# spikes don't self-sustain for so long - need a bit of background activity