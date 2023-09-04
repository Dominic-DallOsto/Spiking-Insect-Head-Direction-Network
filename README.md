# Spiking Attractor Networks as a Model of Insect Head Direction Circuits

We extend the spiking network model from [Diehl and Cook, 2015](https://doi.org/10.3389/fncom.2015.00099) to implement spiking attractor networks with small numbers of neurons, consistent with recent work from [Pisokas et. al, 2020](https://doi.org/10.7554/eLife.53985) that insect head direction circuits can be modelled as spiking networks with 8 neural columns.

## Running the simulations

Required libraries:

- brian2
- pycma

Interesting notebooks:

- [Attractor network based on the Diehl and Cook model](./attractor_network.py)
- [Using CMA-ES to learn different bump shapes](./calibrate_network_bumps.ipynb)
- [Developing a rate based theory for head direction networks](./rate_analysis.ipynb)
- [Using this theory to calibrate spiking networks](./apply%20rate%20analysis%20to%20spiking%20network.ipynb)
