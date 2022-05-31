import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

# b2.set_device('cpp_standalone', build_on_run=False)

b2.start_scope()

# Diehl and Cook neuron model from https://github.com/sdpenguin/Brian2STDPMNIST/blob/2d935d0a98b6c94cfbc1cb8304f16a578a57342b/Diehl%26Cook_spiking_MNIST_Brian2.py#L204
n_e = 16
n_i = 1
v_rest_e = -65. * b2.mV
v_rest_i = -60. * b2.mV
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV
v_thresh_i = -40. * b2.mV
refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

scr_e = 'v = v_reset_e; timer = 0*ms'
offset = 20.0*b2.mV
# v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
v_thresh_e_str = 'v>v_thresh_e'
v_thresh_i_str = 'v>v_thresh_i'
v_reset_i_str = 'v=v_reset_i'


neuron_eqs_e = '''
dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
I_synE = ge * nS *         -v                           : amp
I_synI = gi * nS * (-100.*mV-v)                          : amp
dge/dt = -ge/(5.0*ms)                                   : 1
dgi/dt = -gi/(2.0*ms)                                  : 1
theta      :volt
'''

neuron_eqs_i = '''
dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
I_synE = ge * nS *         -v                           : amp
I_synI = gi * nS * (-85.*mV-v)                          : amp
dge/dt = -ge/(1.0*ms)                                   : 1
dgi/dt = -gi/(2.0*ms)                                  : 1
'''

neuron_group_e = b2.NeuronGroup(n_e, neuron_eqs_e, threshold=v_thresh_e_str, refractory=refrac_e, reset=scr_e, method='euler')
neuron_group_i = b2.NeuronGroup(n_i, neuron_eqs_i, threshold=v_thresh_i_str, refractory=refrac_i, reset=v_reset_i_str, method='euler')
neuron_group_e.v = v_rest_e
neuron_group_i.v = v_rest_i

monitor_e = b2.SpikeMonitor(neuron_group_e)
monitor_i = b2.SpikeMonitor(neuron_group_i)
voltage_e = b2.StateMonitor(neuron_group_e, 'v', record=True)

ee_connection = b2.Synapses(neuron_group_e, neuron_group_e, 'w : 1', on_pre='ge_post += w')
ee_connection.connect(True)
ee_connection.w = '15.0*exp(-(i_pre-i_post)**2 / (2*1.5**2))'

ei_connection = b2.Synapses(neuron_group_e, neuron_group_i, 'w : 1', on_pre='ge_post += w')
ei_connection.connect(True)
ei_connection.w = '2.0'

ie_connection = b2.Synapses(neuron_group_i, neuron_group_e, 'w : 1', on_pre='gi_post += w')
ie_connection.connect(True)
ie_connection.w = '50.0'

plot_weights = False
if plot_weights:
	weights = np.zeros((n_e,n_e))
	for pre, post, weight in zip(ee_connection.i_pre[:], ee_connection.i_post[:], ee_connection.w[:]):
		weights[pre,post] = weight
	plt.imshow(weights)
	plt.show()

neuron_group_input = b2.PoissonGroup(n_e, 0*b2.Hz)
input_connection = b2.Synapses(neuron_group_input, neuron_group_e, 'w : 1', on_pre='ge_post += w')
input_connection.connect('i==j')
input_connection.w = '20.0'
monitor_input = b2.SpikeMonitor(neuron_group_input)

duration = 200*b2.ms

neuron_group_input.rates = [100*b2.Hz if 8<=i<=10 else 70*b2.Hz for i in range(n_e)]
b2.run(duration)
neuron_group_input.rates = [100*b2.Hz if 1<=i<=4 else 20*b2.Hz for i in range(n_e)]
b2.run(duration)
neuron_group_input.rates = 0
b2.run(duration)

# b2.device.build(directory='output', compile=True, run=True, debug=False)

fig, axs = plt.subplots(3, 1, figsize=(12,4), sharex=True)
assert(isinstance(axs, np.ndarray))
axs[0].plot(monitor_e.t/b2.ms, monitor_e.i[:], '.b')
axs[0].set_ylabel('Neuron index')
axs[0].set_ylim([0,n_e])
axs[0].set_title('E spikes')
axs[1].plot(monitor_i.t/b2.ms, monitor_i.i[:], '.r')
axs[1].set_ylabel('Neuron index')
axs[1].set_title('I spikes')
# plt.plot(voltage_e.t/b2.ms, voltage_e.v[:].T)
# plt.set_xlabel('Time (ms)')
# plt.set_ylabel('Neuron index')
# plt.set_title('E voltage')
axs[2].plot(monitor_input.t/b2.ms, monitor_input.i[:], '.k')
axs[2].set_xlabel('Time (ms)')
axs[2].set_ylabel('Neuron index')
axs[2].set_title('Input spikes')
plt.tight_layout()
plt.show()
