#!/usr/bin/env python

# Save file Options
default_dir = './'					# Default dir if scripts are called without dir
weights_file = 'weights.h5'					# Trained weights
training_file = 'training_data.h5'			# Results from training
evaluation_file = 'evaluation_data.h5'		# Results from evaluation

# Network parameters
input_layer_size = 3
output_layer_size = 3				# Left & Forward & Right neuron
left_neuron = 0
forward_neuron = 1
right_neuron = 2

sim_time_step = 50.0				# Length of network simulation during each step in ms
V_reset = -70.						# Reset pontential of the membrane in mV
t_ref = 2.							# Refractory period in ms
time_resolution = 0.01				# Network simulation time resolution in ms
iaf_params = {}						# IAF neuron parameters
poisson_params = {}					# Poisson neuron parameters
max_poisson_freq = 1000.			# Maximum Poisson firing frequency for n_max in Hz
n_max = float(sim_time_step//t_ref)	# Maximum input activity
nest_kernel_status = {				# Nest Kernel initialization options
	"local_num_threads": 1,			# Number of Threads used by nest
	"resolution": time_resolution
}

# R-STDP parameters
w_min = 0.							# Minimum weight value in mV
w_max = 3000.						# Maximum weight value in mV
w0_min = 1500.						# Minimum initial random value in mV
w0_max = 1501.						# Maximum initial random value in mV
# These tau_n and tau_c parameters are suggested by Izhikevich, E.M. (2007). Solving the distal reward problem
# through linkage of STDP and dopamine signaling. Cereb. Cortex, 17(10), 2443-2452.
tau_n = 1.							# Time constant of reward signal in ms
tau_c = 200.						# Time constant of eligibility trace in ms
A_plus = 1.							# Constant scaling strength of potentiation
A_minus = 1.						# Constant scaling strength of depression

r_stdp_synapse_options = {					# Initialisation Options for R-STDP Synapses
	"model": "stdp_dopamine_synapse",		# R-STDP Model
	"weight": {
		"distribution": "uniform",			# Initial weight distribution
		"low": w0_min,
		"high": w0_max
	}
}
