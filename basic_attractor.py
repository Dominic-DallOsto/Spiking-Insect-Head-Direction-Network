import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt
import attractor_network

plot_weights = False
if plot_weights:
	weights = np.zeros((n_e,n_e))
	for pre, post, weight in zip(ee_connection.i_pre[:], ee_connection.i_post[:], ee_connection.w[:]):
		weights[pre,post] = weight
	plt.imshow(weights)
	plt.show()


network = attractor_network.AttractorNetwork()


# network.run_with_inputs(duration, [100*b2.Hz if 1<=i<=4 else 20*b2.Hz for i in range(16)])
# network.run_with_inputs(duration, 0)

# network.run_with_inputs(duration, [100*b2.Hz if 1<=i<=4 else 20*b2.Hz for i in range(16)])

def plot_results(network):
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
	plt.show()

duration = 200*b2.ms
network.run_with_inputs(duration, [100*b2.Hz if 8<=i<=10 else 70*b2.Hz for i in range(network.num_e)])
plot_results(network)

network.reset()
network.params['weight_ee'] = 50
network.run_with_inputs(duration, [100*b2.Hz if 8<=i<=10 else 70*b2.Hz for i in range(network.num_e)])
plot_results(network)

network.reset()
network.params['weight_ie'] = 100
network.run_with_inputs(duration, [100*b2.Hz if 8<=i<=10 else 70*b2.Hz for i in range(16)])
plot_results(network)