import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np

from typing import Callable, Any, List

class ParameterTuner():
	'''Make a plot with sliders for changing parameter values and a button to run a simulation callback.'''
	def __init__(self, params, initialisation_function : Callable[[], Any], run_function_creator: Callable[[Any,List[matplotlib.widgets.Slider]],Callable[[Any],None]]) -> None:
		'''
		params: dictionary of parameters - { name: (min,max,initial) }
		initialisation_function: function to run once, for example to setup a plot for the results
		run_function_creator: function that takes the output of the initialisation function, and sliders, and returns a function to run when button is clicked
		'''
		setup_return = initialisation_function()
		self.setup_slider_menu(params)
		self.button.on_clicked(run_function_creator(setup_return, self.sliders))

	def setup_slider_menu(self, params):
		fig, axs = plt.subplots(len(params)+1, 1, figsize=(4,6))
		assert(isinstance(axs, np.ndarray))
		self.sliders = []
		for (param, (min,max,default)), ax in zip(params.items(),axs):
			slider = matplotlib.widgets.Slider(ax, label=param, valmin=min, valmax=max, valinit=default)
			slider.on_changed(lambda _: None)
			self.sliders.append(slider)
		self.button = matplotlib.widgets.Button(axs[-1], 'Run')
		plt.tight_layout()

	def start(self):
		'''
		Start the parameter tuning (just shows the plots which have been already setup)
		'''
		plt.show()