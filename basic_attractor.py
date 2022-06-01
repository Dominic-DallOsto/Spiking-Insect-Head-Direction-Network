from distutils.core import setup
import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt
import attractor_network
import parameter_tuner

network = attractor_network.AttractorNetwork()

set_unit_value = lambda unit, value: value*unit.get_best_unit() if not b2.is_dimensionless(unit) else value

def run_sim(axs, sliders):
	def run(_):
		network.reset()
		network.params.update({slider.label._text : set_unit_value(network.params[slider.label._text], slider.val) for slider in sliders})
		network.run_with_inputs(duration, [100*b2.Hz if 8<=i<=10 else 40*b2.Hz for i in range(network.num_e)])
		network.run_with_inputs(duration, 0)
		plot_results(network, axs)
	return run

def plot_results(network, axs):
	for ax in axs:
		for plot in ax.lines + ax.collections:
			plot.remove()

	axs[0].plot(network.monitor_e.t/b2.ms, network.monitor_e.i[:], '.b')
	axs[1].plot(network.monitor_i.t/b2.ms, network.monitor_i.i[:], '.r')
	axs[2].plot(network.monitor_input.t/b2.ms, network.monitor_input.i[:], '.k')

	axs[0].get_figure().canvas.draw_idle()

def setup_plot_wrapper(network):
	def setup_plot():
		fig, axs = plt.subplots(3, 1, figsize=(12,4), sharex=True)
		assert(isinstance(axs, np.ndarray))
		axs[0].plot(network.monitor_e.t/b2.ms, network.monitor_e.i[:], '.b')
		axs[0].set_ylabel('Neuron index')
		axs[0].set_ylim([0,network.num_e])
		axs[0].set_title('E spikes')
		axs[1].plot(network.monitor_i.t/b2.ms, network.monitor_i.i[:], '.r')
		axs[1].set_ylabel('Neuron index')
		axs[1].set_title('I spikes')
		axs[2].plot(network.monitor_input.t/b2.ms, network.monitor_input.i[:], '.k')
		axs[2].set_xlabel('Time (ms)')
		axs[2].set_ylabel('Neuron index')
		axs[2].set_title('Input spikes')
		plt.tight_layout()

		return axs
	return setup_plot


duration = 200*b2.ms

params_to_tune = {
	'weight_ee': (0, 100, 15),
	'weight_ei': (0, 10, 2),
	'weight_ie': (0, 100, 50),
	'weight_input': (0, 100, 20),
	'tau_e': (0,10,5)
}
parameter_tuner = parameter_tuner.ParameterTuner(params_to_tune, setup_plot_wrapper(network), run_sim)
parameter_tuner.start()
