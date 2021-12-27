import os

from collections import OrderedDict
import time
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_netpyne.config import *

# TODO: which values should go here?
w_E_E_prox = 0.0
w_E_I_prox = 0.01
w_I_E = 0.01
w_I_I = 0.0045

work_path = os.getcwd()
data_path = os.path.join(work_path.split("tvb_netpyne")[0], "data")
outputs_path = os.path.join(work_path, "netpyne-outputs/RWW-coupled-EE-II-0.0075-EI-0.01-IE-0.04")
config = Config(output_base=outputs_path)

config.figures.SHOW_FLAG = True
config.figures.SAVE_FLAG = True
config.figures.FIG_FORMAT = 'png'
config.figures.DEFAULT_SIZE= config.figures.NOTEBOOK_SIZE
FIGSIZE = config.figures.DEFAULT_SIZE

from tvb_multiscale.core.plot.plotter import Plotter
plotter = Plotter(config.figures)

# For interactive plotting:
# %matplotlib notebook  

# Otherwise:
# %matplotlib inline 

##################################################################################################################
# 1. Load structural data (minimally a TVB connectivity) & prepare TVB simulator (region mean field model, integrator, monitors etc)
##################################################################################################################

from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorBuilder

# # ----------------------1. Build a TVB simulator (model, integrator, monitors...)----------------------------------
# simulator_builder = CoSimulatorBuilder()
# simulator_builder.connectivity = config.DEFAULT_CONNECTIVITY_ZIP
# simulator_builder.model = ReducedWongWangExcIOInhI
# model_params = {"G": np.array([2.0, ]), "lamda": np.array([0., ]), 
#                 "w_p": np.array([1.4, ]), "J_i": np.array([1.0, ])}
# simulator = simulator_builder.build(**model_params)


# Optionally modify the default configuration:

# ----------------------------------------------------------------------------------------------------------------
# ----Uncomment below to modify the simulator by changing the default options:--------------------------------------
# ----------------------------------------------------------------------------------------------------------------

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.cosimulator import CoSimulator
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.monitors import Raw  # , Bold, EEG
    
# Load connectivity
connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
# Normalize connectivity weights
connectivity.weights = connectivity.scaled_weights(mode="region")
connectivity.weights /= np.percentile(connectivity.weights, 95)
connectivity.weights[connectivity.weights>1.0] = 1.0
connectivity.configure()

# Create a TVB simulator and set all desired inputs
# (connectivity, model, surface, stimuli etc)
# We choose all defaults in this example
simulator = CoSimulator()
simulator.model = ReducedWongWangExcIOInhI()
simulator.model.G = np.array([2.0, ])     # Global cloupling scaling
simulator.model.lamda = np.array([0.2, ]) # Feedforward inhibition
simulator.model.w_p = np.array([1.4, ])   # Feedback excitation
simulator.model.J_i = np.array([1.0, ])   # Feedback inhibition

simulator.connectivity = connectivity

simulator.integrator = HeunStochastic()
simulator.integrator.dt = 0.1
simulator.integrator.noise.nsig = np.array([1e-6])

# Setting initial conditions
# convert time delays to integration step delays:
simulator.connectivity.set_idelays(simulator.integrator.dt) 
# determine the history horizon (past states buffer)
simulator.horizon = simulator.connectivity.idelays.max() + 1
# # The initial conditions must have a shape of 
# # (past_time_horizon, number_of_variables, number_of_regions, number_of_modes)
# init_cond = np.load(os.path.join(config.out.FOLDER_RES, "init_cond.npy"))
# simulator.initial_conditions = init_cond * np.ones((simulator.horizon,
#                                                     simulator.model.nvar,
#                                                     simulator.connectivity.number_of_regions,
#                                                     simulator.model.number_of_modes))

mon_raw = Raw(period=1.0)  # ms
simulator.monitors = (mon_raw, )

# ----------------------------------------------------------------------------------------------------------------
# ----Uncomment below to modify the builder by changing the default options:--------------------------------------
# ----------------------------------------------------------------------------------------------------------------
plotter.plot_tvb_connectivity(simulator.connectivity);


##################################################################################################################
# 2. Build and connect the NetPyNE network model (networks of spiking neural populations for fine-scale regions, stimulation devices, spike recorders etc)
##################################################################################################################

# Select the regions for the fine scale modeling with NetPyNE spiking networks
number_of_regions = simulator.connectivity.region_labels.shape[0]
spiking_nodes_ids = []  # the indices of fine scale regions modeled with NetPyNE
# We model parahippocampal cortices (left and right) with NetPyNE
for id in range(number_of_regions):
    if simulator.connectivity.region_labels[id].find("hippocampal_L") >= 0:
        spiking_nodes_ids.append(id)

# originally - WWDeco2014Builder        
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder

# Build a NetPyNE network model with the corresponding builder
netpyne_network_builder = DefaultExcIOInhIBuilder(simulator, spiking_nodes_ids, config=config)
netpyne_network_builder.configure()

# N_E = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_e)
# N_I = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_i)


# # Using all default parameters for this example
netpyne_network_builder.set_defaults()

# or...

# ----------------------------------------------------------------------------------------------------------------
# ----Uncomment below to modify the builder by changing the default options:--------------------------------------
# ----------------------------------------------------------------------------------------------------------------
from copy import deepcopy

population_neuron_model = "PYR"  # the NetPyNE spiking neuron model # TODO: use same as internally

netpyne_network_builder.population_order = 100

netpyne_network_builder.scale_e = 1.3
netpyne_network_builder.scale_i = 0.6

N_E = int(netpyne_network_builder.population_order * netpyne_network_builder.scale_e)
N_I = int(netpyne_network_builder.population_order * netpyne_network_builder.scale_i)

# TODO: here and below: check if global_coupling_scaling is used properly and not redundantly (both for netpyne_model_builder and tvb_netpyne_builder)
netpyne_network_builder.global_coupling_scaling = \
    netpyne_network_builder.tvb_serial_sim["coupling.a"][0].item() * \
    netpyne_network_builder.tvb_serial_sim["model.G"][0].item()
netpyne_network_builder.lamda = netpyne_network_builder.tvb_serial_sim["model.lamda"][0].item()


# When any of the properties model, params and scale below depends on regions,
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property

def param_fun(node_index, params, weight):
    w_E_ext = \
        weight * netpyne_network_builder.tvb_weights[:, node_index]
    w_E_ext[node_index] = 1.0  # this is external input weight to this node
    out_params = deepcopy(params)
    out_params.update({"w_E_ext": w_E_ext})
    return out_params
    

common_params = {
    "V_th": -50.0, "V_reset": -55.0, "E_L": -70.0, "E_ex": 0.0, "E_in": -70.0,                       # mV
    "tau_decay_AMPA": 2.0, "tau_decay_GABA_A": 10.0, "tau_decay_NMDA": 100.0, "tau_rise_NMDA": 2.0,  # ms
    "s_AMPA_ext_max": N_E * np.ones((netpyne_network_builder.number_of_regions,)).astype("f"), 
    "N_E": N_E, "N_I": N_I, "epsilon": 1.0  # /N_E
}
params_E = {
    "C_m": 500.0,    # pF
    "g_L": 25.0,     # nS
    "t_ref": 2.0,    # ms
    "g_AMPA_ext": 3.37, "g_AMPA": 0.065, "g_NMDA": 0.20, "g_GABA_A": 10.94,  # nS
    "w_E": netpyne_network_builder.tvb_serial_sim["model.w_p"][0].item(), 
    "w_I": netpyne_network_builder.tvb_serial_sim["model.J_i"][0].item()
}
params_E.update(common_params)
netpyne_network_builder.params_E = \
    lambda node_index: param_fun(node_index, params_E,
                                 weight=netpyne_network_builder.global_coupling_scaling)

params_I = {
    "C_m": 200.0,  # pF
    "g_L": 20.0,   # nS
    "t_ref": 1.0,  # ms
    "g_AMPA_ext": 2.59, "g_AMPA": 0.051,"g_NMDA": 0.16, "g_GABA_A": 8.51,  # nS
    "w_E": 1.0, "w_I": 1.0
}
params_I.update(common_params)
netpyne_network_builder.params_I = \
    lambda node_index: param_fun(node_index, params_I,
                                 weight=netpyne_network_builder.lamda * netpyne_network_builder.global_coupling_scaling)

# Populations' configurations
# When any of the properties model, params and scale below depends on regions,
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property
netpyne_network_builder.populations = [
    {"label": "E", "model": population_neuron_model,
     "nodes": None,  # None means "all"
     "params": netpyne_network_builder.params_E,
     "scale": netpyne_network_builder.scale_e},
    {"label": "I", "model": population_neuron_model,
     "nodes": None,  # None means "all"
     "params": netpyne_network_builder.params_I,
     "scale": netpyne_network_builder.scale_i}
  ]

# Within region-node connections
# When any of the properties model, conn_spec, weight, delay, receptor_type below
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property
# TODO: be mindful that for NetPyNE it's done different way (weight 1.0 and -1.0 for E and I respectively, and same 'static_synapse')
synapse_model = "synapse_model_placeholder"
receptor_type_E = "exc"
receptor_type_I = "inh"
# conn_spec = {"allow_autapses": True, 'allow_multapses': True, 'rule': "all_to_all",
#              "indegree": None, "outdegree": None, "N": None, "p": 0.1}

# TOOD: Does Netpyne allow autapses/multapses? Does all_to_all equals to prob=1.0 in Netpyne?
conn_spec_all_to_all = {"rule": "all_to_all"}
conn_spec_prob_low = {"rule": {"prob": 0.1}}
conn_spec_prob_high = {"rule": {"prob": 0.5}}

within_node_delay = 5
#                     {"distribution": "uniform", 
#                      "low": np.minimum(netpyne_model_builder.default_min_delay, 
#                                        netpyne_model_builder.default_populations_connection["delay"]), 
#                      "high": np.maximum(netpyne_model_builder.tvb_dt, 
#                                         2*netpyne_model_builder.default_populations_connection["delay"])}

# connections between populations within same spiking region
netpyne_network_builder.populations_connections = [
     #              ->
    {"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
     "synapse_model": synapse_model, "conn_spec": conn_spec_prob_low,
     "weight": w_E_E_prox, "delay": within_node_delay,
     "receptor_type": receptor_type_E, "nodes": None},  # None means apply to all
    {"source": "E", "target": "I",  # E -> I
     "synapse_model": synapse_model, "conn_spec": conn_spec_prob_high,
     "weight": w_E_I_prox, "delay": within_node_delay,
     "receptor_type": receptor_type_E, "nodes": None},  # None means apply to all
    {"source": "I", "target": "E",  # I -> E
     "synapse_model": synapse_model, "conn_spec": conn_spec_all_to_all, 
     "weight": w_I_E, "delay": within_node_delay, # TODO: 1:23:02 says: -nest_model_builder.tvb_model.J_i[0].item
     "receptor_type": receptor_type_I, "nodes": None},  # None means apply to all
    {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
     "synapse_model": synapse_model, "conn_spec": conn_spec_prob_low,
     "weight": w_I_I, "delay": within_node_delay,
     "receptor_type": receptor_type_I, "nodes": None}  # None means apply to all
    ]


# Among/Between region-node connections
# Given that only the AMPA population of one region-node couples to
# all populations of another region-node,
# we need only one connection type
        
# When any of the properties model, conn_spec, weight, delay, receptor_type below
# depends on regions, set a handle to a function with
# arguments (source_region_index=None, target_region_index=None)

from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay, tvb_weight, scale_tvb_weight
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import random_uniform_tvb_delay
    
lamda = netpyne_network_builder.tvb_serial_sim["model.lamda"][0]
tvb_to_netpyne_weight_scaling = 10.0 # This value doesn't seem to have any theoretical meaning in itself. Rather it should be picked up by trial and error to make synaptic strength of connections between spiking nodes and mean-field nodes be biologically plausible.

# from 1:24:12
tvb_weight_fun = lambda source_node, target_node: \
                scale_tvb_weight(source_node, target_node, netpyne_network_builder.tvb_weights, scale = tvb_to_netpyne_weight_scaling * netpyne_network_builder.global_coupling_scaling)

tvb_delay_fun = lambda source_node, target_node: \
                 tvb_delay(source_node, target_node, netpyne_network_builder.tvb_delays)
#                  random_uniform_tvb_delay(source_node, target_node, netpyne_model_builder.tvb_delays, 
#                                           low=netpyne_model_builder.tvb_dt, 
#                                           high=2*netpyne_model_builder.tvb_dt, 
#                                           sigma=0.1)

# connections between populations between different spiking regions
# Total excitatory spikes of one region node will be distributed to
netpyne_network_builder.nodes_connections = [
    #              ->
    {"source": "E", "target": ["E"],
     "synapse_model": synapse_model, "conn_spec": conn_spec_all_to_all,
     "weight": tvb_weight_fun, "delay": tvb_delay_fun,  
    # Each region emits spikes in its own port:
     "receptor_type": receptor_type_E,
     "source_nodes": None, "target_nodes": None}  # None means apply to all
    ]

if lamda > 0:
    tvb_weight_fun_E_I = lambda source_node, target_node: \
                scale_tvb_weight(source_node, target_node, netpyne_network_builder.tvb_weights, scale=lamda * netpyne_network_builder.global_coupling_scaling)

    netpyne_network_builder.nodes_connections.append(
        {"source": "E", "target": ["I"],
         "synapse_model": synapse_model, "conn_spec": conn_spec_all_to_all,
         "weight": tvb_weight_fun_E_I, "delay": tvb_delay_fun,  
        # Each region emits spikes in its own port:
         "receptor_type": receptor_type_E,
         "source_nodes": None, "target_nodes": None}  # None means apply to all
    )
    
# Creating  devices to be able to observe NetPyNE activity:

netpyne_network_builder.output_devices = []

connections = OrderedDict({})
#          label <- target population
connections["E_spikes"] = "E"
connections["I_spikes"] = "I"
netpyne_network_builder.output_devices.append(
    {"model": "spike_recorder", "params": {"record_to": "memory"},
     "connections": connections, "nodes": None})  # None means apply to all

# Labels have to be different

#TODO: multimeter should go here
    
# Create a spike stimulus input device
# TODO: Background noisy stimulus. Commented out temporarily for clearer debugging.
# netpyne_network_builder.input_devices = [
#     {"model": "poisson_generator",
#      "params": {"rate": 7200.0, "origin": 0.0, "start": 0.1},  # "stop": 100.0
#      "connections": {"Stimulus": ["E", "I"]}, 
#      "nodes": None,         # None means apply to all
#      "weights": 1.0 * tvb_to_netpyne_weight_scaling, 
#      "delays": netpyne_network_builder.tvb_dt, 
# #      {"distribution": "uniform", 
# #                 "low": netpyne_model_builder.tvb_dt, 
# #                 "high": 2*netpyne_model_builder.tvb_dt},
#     "receptor_type": receptor_type_E},
#                                   ]  #

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

tvb_to_netpyne_state_variable = "R_e" # TODO: better name?
# Due to peculiarities in timing of initialization of NetPyNE network, one should take care of creating stimuli population in advance, prior to building interface and creating connections (next step)

netpyne_network_builder.state_variable = tvb_to_netpyne_state_variable
# netpyne_network_builder.populations_to_couple_state_variable_to = lamda > 0

netpyne_network = netpyne_network_builder.build(set_defaults=False) # or true, if netpyne_model_builder.set_default() is not yet ran

# print(netpyne_network.print_str(connectivity=True))

##################################################################################################################
# 3. Build the TVB-NetPyNE interface
##################################################################################################################

from tvb_multiscale.tvb_netpyne.interfaces.builders.models.default import DefaultInterfaceBuilder 

# Build a TVB-NetPyNE interface with all the appropriate connections between the
# TVB and NetPyNE modelled regions
tvb_netpyne_builder = \
    DefaultInterfaceBuilder(simulator, netpyne_network, spiking_nodes_ids, 
                            exclusive_nodes=True, populations_sizes=[N_E, N_I])

tvb_to_netpyne_mode = "rate"  # TVB also has "current" and "param" options, but they aren't yet implemented in NetPyNE
netpyne_to_tvb = True

# Using all default parameters for this example

# or...


# ----------------------------------------------------------------------------------------------------------------
# ----Uncomment below to modify the builder by changing the default options:--------------------------------------
# ----------------------------------------------------------------------------------------------------------------

from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay, scale_tvb_weight
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates \
    import random_normal_tvb_weight, random_uniform_tvb_delay, receptor_by_source_region

lamda = netpyne_network_builder.tvb_serial_sim["model.lamda"][0].item()
G = netpyne_network_builder.tvb_serial_sim["model.G"][0].item()
# tvb_netpyne_builder.global_coupling_scaling = G * netpyne_network_builder.tvb_serial_sim["coupling.a"][0].item()
tvb_netpyne_builder.global_coupling_scaling = netpyne_network_builder.global_coupling_scaling # this is what video says. original version is commented out above

# TVB -> NetPyNE

if tvb_to_netpyne_mode in ["rate", "current"]:

    tvb_weight_fun = lambda tvb_node_id, netpyne_node_id: \
                        scale_tvb_weight(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_weights, scale=tvb_to_netpyne_weight_scaling * tvb_netpyne_builder.global_coupling_scaling)
    tvb_delay_fun = lambda tvb_node_id, netpyne_node_id: \
                        tvb_delay(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_delays)
#                         random_uniform_tvb_delay(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_delays,
#                                                  low=tvb_netpyne_builder.tvb_dt, 
#                                                  high=2*tvb_netpyne_builder.tvb_dt,
#                                                  sigma=0.1)

# --------For spike transmission from TVB to NetPyNE devices acting as TVB proxy nodes with TVB delays:--------

# Mean spike rates are applied in parallel to all target neurons

if tvb_to_netpyne_mode == "rate":
    weights = tvb_weight_fun # {"distribution": "normal", "mu": 1.0, "sigma": 0.1}
    
    tvb_netpyne_builder.tvb_to_spikeNet_interfaces = [
        {"model": "poisson_generator",
         "params": {"connectivity_scale": 1.0},
    # # ---------Properties potentially set as function handles with args (tvb_node_id=None)-------------------------
         "interface_weights": 1.0 * N_E, # Convert mean value to total value
    # Applied outside NetPyNE for each interface device
    # -------Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)-----------
        "weights": weights, "delays": tvb_delay_fun,
        "receptor_type": receptor_type_E,
        # --------------------------------------------------------------------------------------------------------------
        #             TVB sv -> NetPyNE population
        "connections": {tvb_to_netpyne_state_variable: ["E"]},
        "source_nodes": None, "target_nodes": None}]  # None means all here

    if lamda > 0.0:
        tvb_netpyne_builder.tvb_to_spikeNet_interfaces.append(
            {"model": "poisson_generator",
             "params": {"connectivity_scale": lamda},
        # # ---------Properties potentially set as function handles with args (tvb_node_id=None)-------------------------
             "interface_weights": lamda * N_E, # Convert mean value to total value
        # Applied outside NetPyNE for each interface device
        # -------Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)-----------
            "weights": weights, "delays": tvb_delay_fun,
            "receptor_type": receptor_type_E,
            # --------------------------------------------------------------------------------------------------------------
            #             TVB sv -> NetPyNE population
            "connections": {tvb_to_netpyne_state_variable: ["I"]},
            "source_nodes": None, "target_nodes": None
            }
        )

if netpyne_to_tvb:
    # NetPyNE -> TVB:
    # Use S_e and S_i instead of r_e and r_i
    # for transmitting to the TVB state variables directly
    connections = OrderedDict()
    #            TVB <- NetPyNE
    connections["Rin_e"] = ["E"]
    connections["Rin_i"] = ["I"]
    tvb_netpyne_builder.spikeNet_to_tvb_interfaces = [
        {"model": "spike_recorder", "params": {},
    # ------------------Properties potentially set as function handles with args (netpyne_node_id=None)--------------------
         "interface_weights": 1.0, "delays": 0.0,
    # -----------------------------------------------------------------------------------------------------------------
         "connections": connections, "nodes": None}]  # None means all here

    
tvb_netpyne_builder.w_tvb_to_current = 1000 * netpyne_network_builder.tvb_serial_sim["model.J_N"][0]  # (nA of TVB -> pA of NetPyNE)
# WongWang model parameter r is in Hz, just like poisson_generator assumes in NetPyNE:
tvb_netpyne_builder.w_tvb_to_spike_rate = 1.0
# We return from a NetPyNE spike_detector the ratio number_of_population_spikes / number_of_population_neurons
# for every TVB time step, which is usually a quantity in the range [0.0, 1.0],
# as long as a neuron cannot fire twice during a TVB time step, i.e.,
# as long as the TVB time step (usually 0.001 to 0.1 ms)
# is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
# For conversion to a rate, one has to do:
# w_spikes_to_tvb = 1/tvb_dt, to get it in spikes/ms, and
# w_spikes_to_tvb = 1000/tvb_dt, to get it in Hz
# given WongWang model parameter r is in Hz but tvb dt is in ms:
tvb_netpyne_builder.w_spikes_to_tvb = 1000.0 / tvb_netpyne_builder.tvb_dt
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
tvb_netpyne_model = tvb_netpyne_builder.build_interface(tvb_to_spikeNet_mode=tvb_to_netpyne_mode, spikeNet_to_tvb=netpyne_to_tvb)

# print(tvb_nest_model.print_str(detailed_output=True, connectivity=False))

# for ii in range(len(tvb_netpyne_builder.tvb_to_spikeNet_interfaces)):
#     print("Interface=%d" % ii)
#     for key in tvb_netpyne_builder.tvb_to_spikeNet_interfaces[ii].keys():
#         print(key)
#         print(tvb_netpyne_builder.tvb_to_spikeNet_interfaces[ii][key])

##################################################################################################################
# 4. Configure simulator, simulate, gather results
##################################################################################################################

# Configure the simulator with the TVB-NetPyNE interface...
simulator.configure(tvb_spikeNet_interface=tvb_netpyne_model)
# ...and simulate!
t = time.time()
simulation_length=40.0  # Set at least 1100.0 for a meaningful simulation
transient = simulation_length/11
results = simulator.run(simulation_length=simulation_length)

# TODO: is below ever needed?
# Integrate NetPyNE one more NetPyNE time step so that multimeters get the last time point
# unless you plan to continue simulation later
# simulator.run_spiking_simulator(simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
print("\nSimulated in %f secs!" % (time.time() - t))

# Clean-up NetPyNE simulation
# simulator.tvb_spikeNet_interface.nest_instance.Cleanup()

##################################################################################################################
# 5. Plot results and write them to HDF5 files
##################################################################################################################

# set to False for faster plotting of only mean field variables and dates, apart from spikes" rasters:
from scipy.io import savemat
plot_per_neuron = False 
MAX_VARS_IN_COLS = 3
MAX_REGIONS_IN_ROWS = 10
MIN_REGIONS_FOR_RASTER_PLOT = 9
# from examples.plot_write_results import plot_write_results
# populations = []
# populations_sizes = []
# for pop in netpyne_model_builder.populations:
#     populations.append(pop["label"])
#     populations_sizes.append(int(np.round(pop["scale"] * netpyne_model_builder.population_order)))
# plot_write_results(results, simulator, populations=populations, populations_sizes=populations_sizes, 
#                    transient=transient, tvb_state_variable_type_label="State Variables", 
#                    tvb_state_variables_labels=simulator.model.variables_of_interest, 
#                    plot_per_neuron=plot_per_neuron, plotter=plotter, config=config)

# If you want to see what the function above does, take the steps, one by one
try:
    # We need framework_tvb for writing and reading from HDF5 files
    from tvb_multiscale.tvb.core.io.h5_writer import H5Writer
    writer = H5Writer()
except:
    writer = False
    
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

# Put the results in a Timeseries instance
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion

source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
        data=results[0][1], time=results[0][0]-results[0][0][0],
        connectivity=simulator.connectivity,
        labels_ordering=["Time", "State Variable", "Region", "Neurons"],
        labels_dimensions={"State Variable": list(simulator.model.variables_of_interest),
                           "Region": simulator.connectivity.region_labels.tolist()},
        sample_period=simulator.integrator.dt)
source_ts.configure()

t = source_ts.time
    
# np.save(os.path.join(config.out.FOLDER_RES, "init_cond.npy"), results[0][1][-100:].mean(axis=0))

# Write to file
if writer:
    writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(source_ts._data,
                                                                    connectivity=source_ts.connectivity),
                           os.path.join(config.out.FOLDER_RES, source_ts.title)+".h5")
# uncomment to print it
# source_ts

# Plot TVB time series
# source_ts.plot_timeseries(plotter_config=plotter.config, 
#                           hue="Region" if source_ts.shape[2] > MAX_REGIONS_IN_ROWS else None, 
#                           per_variable=source_ts.shape[1] > MAX_VARS_IN_COLS, 
#                           figsize=FIGSIZE);

# TVB time series raster plot:
# if source_ts.number_of_labels > MIN_REGIONS_FOR_RASTER_PLOT:
#     source_ts.plot_raster(plotter_config=plotter.config, 
#                           per_variable=source_ts.shape[1] > MAX_VARS_IN_COLS,
#                           figsize=FIGSIZE);

# Focus on the nodes modelled in NetPyNE: 
n_spiking_nodes = len(simulator.tvb_spikeNet_interface.spiking_nodes_inds)
source_ts_spiking = source_ts[:, :, simulator.tvb_spikeNet_interface.spiking_nodes_inds]
source_ts_spiking.plot_timeseries(plotter_config=plotter.config, 
                               hue="Region" if source_ts_spiking.shape[2] > MAX_REGIONS_IN_ROWS else None, 
                               per_variable=source_ts_spiking.shape[1] > MAX_VARS_IN_COLS, 
                               figsize=FIGSIZE, figname="Spiking nodes TVB Time Series");

# Focus on the nodes modelled in NetPyNE: raster plot
if source_ts_spiking.number_of_labels > MIN_REGIONS_FOR_RASTER_PLOT:
    source_ts_spiking.plot_raster(plotter_config=plotter.config, 
                               per_variable=source_ts_spiking.shape[1] > MAX_VARS_IN_COLS,
                               figsize=FIGSIZE, figname="Spiking nodes TVB Time Series Raster");

### Spiking Network plots

from tvb_multiscale.core.data_analysis.spiking_network_analyser import SpikingNetworkAnalyser
# Create a SpikingNetworkAnalyzer:
spikeNet_analyzer = \
    SpikingNetworkAnalyser(spikeNet=netpyne_network,
                           start_time=source_ts.time[0], end_time=source_ts.time[-1], 
                           period=simulator.monitors[0].period, transient=transient,
                           time_series_output_type="TVB", return_data=True, 
                           force_homogeneous_results=True, connectivity=simulator.connectivity)

### Plot spikes' raster and mean spike rates and correlations

# Spikes rates and correlations per Population and Region
spikes_res = \
    spikeNet_analyzer.\
        compute_spikeNet_spikes_rates_and_correlations(
            populations_devices=None, regions=None,
            rates_methods=[], rates_kwargs=[{}], rate_results_names=[],
            corrs_methods=[], corrs_kwargs=[{}], corrs_results_names=[], bin_kwargs={},
            data_method=spikeNet_analyzer.get_spikes_from_device, data_kwargs={},
            return_devices=False
        );

if spikes_res:
    print(spikes_res["mean_rate"])
    print(spikes_res["spikes_correlation_coefficient"])
    # Plot spikes' rasters together with mean population's spikes' rates' time series
    if plotter:
        plotter.plot_spike_events(spikes_res["spikes"], rates=spikes_res["mean_rate_time_series"], figsize=FIGSIZE)
        from tvb_multiscale.core.plot.correlations_plot import plot_correlations
        plot_correlations(spikes_res["spikes_correlation_coefficient"], plotter)

if spikes_res:
    print("Mean spike rates:")
    for pop in spikes_res["mean_rate"].coords["Population"]:
        for reg in spikes_res["mean_rate"].coords["Region"]:
            if not np.isnan(spikes_res["mean_rate"].loc[pop, reg]):
                print("%s - %s: %g" % (pop.values.item().split("_spikes")[0], reg.values.item(), 
                                       spikes_res["mean_rate"].loc[pop, reg].values.item()))

    # savemat(os.path.join(config.out.FOLDER_RES, "spikes_mean_rates.mat"), spikes_res["mean_rate"].to_dict())

spikeNet_analyzer.resample = True
spikes_sync = \
    spikeNet_analyzer.compute_spikeNet_synchronization(populations_devices=None, regions=None,
                                                       comp_methods=[spikeNet_analyzer.compute_spikes_sync, 
                                                                     spikeNet_analyzer.compute_spikes_sync_time_series, 
                                                                     spikeNet_analyzer.compute_spikes_distance, 
                                                                     spikeNet_analyzer.compute_spikes_distance_time_series,
                                                                     spikeNet_analyzer.compute_spikes_isi_distance, 
                                                                     spikeNet_analyzer.compute_spikes_isi_distance_time_series],
                                                       computations_kwargs=[{}], data_kwargs={},
                                                       return_spikes_trains=False, return_devices=False)
# print(spikes_sync)

if spikes_sync:
    plotter.config.FONTSIZE = 20 # plotter.config.LARGE_FONTSIZE  # LARGE = 12, default = 10
    plotter.plot_spike_events(spikes_res["spikes"], 
                              time_series=spikes_sync["spikes_sync_time_series"], 
                              mean_results=spikes_sync["spikes_sync"], 
                              plot_spikes=True, spikes_alpha=0.25,
                              spikes_markersize=1.0, time_series_marker="*", 
                              figsize=(50, 7), n_y_ticks=3, n_time_ticks=4, show_time_axis=True,
                              time_axis_min=0.0, time_axis_max=simulation_length
                                     )

if spikes_sync:
    plotter.config.FONTSIZE = 20 # plotter.config.LARGE_FONTSIZE  # LARGE = 12, default = 10
    plotter.plot_spike_events(spikes_res["spikes"], 
                              time_series=spikes_sync["spikes_distance_time_series"], 
                              mean_results=spikes_sync["spikes_distance"], 
                              plot_spikes=True, spikes_alpha=0.25,
                              spikes_markersize=1.0, time_series_marker="*", 
                              figsize=(50, 7), n_time_ticks=4, show_time_axis=True, n_y_ticks=4,
                              time_axis_min=0.0, time_axis_max=simulation_length
                                     )

if spikes_sync:
    plotter.config.FONTSIZE = 20 # plotter.config.LARGE_FONTSIZE  # LARGE = 12, default = 10
    plotter.plot_spike_events(spikes_res["spikes"], 
                              time_series=spikes_sync["spikes_isi_distance_time_series"], 
                              mean_results=spikes_sync["spikes_isi_distance"], 
                              plot_spikes=True, spikes_alpha=0.25,
                              spikes_markersize=1.0,  time_series_marker="*", 
                              figsize=(50, 7), n_y_ticks=3, n_time_ticks=4, show_time_axis=True,
                              time_axis_min=0.0, time_axis_max=simulation_length
                                     )

if spikes_sync:
    print("Spike synchronization:")
    for pop in spikes_sync["spikes_sync"].coords["Population"]:
        for reg in spikes_sync["spikes_sync"].coords["Region"]:
            if not np.isnan(spikes_sync["spikes_sync"].loc[pop, reg]):
                print("%s - %s: %g" % (pop.values.item().split("_spikes")[0], reg.values.item(), 
                                       spikes_sync["spikes_sync"].loc[pop, reg].values.item()))

#     savemat(os.path.join(config.out.FOLDER_RES, "spikes_sync.mat"), spikes_sync["spikes_sync"].to_dict())
#     savemat(os.path.join(config.out.FOLDER_RES, "spikes_sync_time_series.mat"), spikes_sync["spikes_sync_time_series"].to_dict())

if spikes_sync:
    print("Spike distance:")
    for pop in spikes_sync["spikes_distance"].coords["Population"]:
        for reg in spikes_sync["spikes_distance"].coords["Region"]:
            if not np.isnan(spikes_sync["spikes_distance"].loc[pop, reg]):
                print("%s - %s: %g" % (pop.values.item().split("_spikes")[0], reg.values.item(), 
                                       spikes_sync["spikes_distance"].loc[pop, reg].values.item()))

#     savemat(os.path.join(config.out.FOLDER_RES, "spikes_distance.mat"), spikes_sync["spikes_distance"].to_dict())
#     savemat(os.path.join(config.out.FOLDER_RES, "spikes_distance_time_series.mat"), spikes_sync["spikes_distance_time_series"].to_dict())

if spikes_sync:
    print("Spike ISI distance:")
    for pop in spikes_sync["spikes_isi_distance"].coords["Population"]:
        for reg in spikes_sync["spikes_isi_distance"].coords["Region"]:
            if not np.isnan(spikes_sync["spikes_isi_distance"].loc[pop, reg]):
                print("%s - %s: %g" % (pop.values.item().split("_spikes")[0], reg.values.item(), 
                                       spikes_sync["spikes_isi_distance"].loc[pop, reg].values.item()))

#     savemat(os.path.join(config.out.FOLDER_RES, "spikes_isi_distance.mat"), spikes_sync["spikes_isi_distance"].to_dict())
#     savemat(os.path.join(config.out.FOLDER_RES, "spikes_isi_distance_time_series.mat"), spikes_sync["spikes_isi_distance_time_series"].to_dict())

if spikes_res and writer:
    writer.write_object(spikes_res["spikes"].to_dict(), 
                        path=os.path.join(config.out.FOLDER_RES,  "Spikes") + ".h5");
    writer.write_object(spikes_res["mean_rate"].to_dict(),
                        path=os.path.join(config.out.FOLDER_RES,
                                          spikes_res["mean_rate"].name) + ".h5");
    writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(
                              spikes_res["mean_rate_time_series"]._data,
                               connectivity=spikes_res["mean_rate_time_series"].connectivity),
                           os.path.join(config.out.FOLDER_RES,
                                        spikes_res["mean_rate_time_series"].title) + ".h5",
                           recursive=False);
    writer.write_object(spikes_res["spikes_correlation_coefficient"].to_dict(),
                        path=os.path.join(config.out.FOLDER_RES,
                                          spikes_res["spikes_correlation_coefficient"].name) + ".h5");

### Get SpikingNetwork mean field variable time series and plot them

# Continuous time variables' data of spiking neurons
if plot_per_neuron:
    spikeNet_analyzer.return_data = True
else:
    spikeNet_analyzer.return_data = False
spikeNet_ts = \
    spikeNet_analyzer. \
         compute_spikeNet_mean_field_time_series(populations_devices=None, regions=None, variables=None,
                                                 computations_kwargs={}, data_kwargs={}, return_devices=False)
if spikeNet_ts:
    if plot_per_neuron:
        mean_field_ts = spikeNet_ts["mean_field_time_series"]  # mean field
        spikeNet_ts = spikeNet_ts["data_by_neuron"]  # per neuron data
    else:
        mean_field_ts = spikeNet_ts
        spikeNet_ts = None
    if mean_field_ts and mean_field_ts.size > 0:
        # Compute total sum of external synapses time series:
        mean_field_ext_mean = mean_field_ts[:, :3]
        for ilbl, lbl in enumerate(["spikes_exc_ext_0", "s_AMPA_ext_0", "I_AMPA_ext_0"]):
            mean_field_ext_mean._data[:, ilbl] = mean_field_ts[:, lbl::3].sum(axis=1)
        mean_field_ext_mean._data.name = "Total External Synapses' Mean Field Time Series"
        coords = dict(mean_field_ext_mean.coords)
        coords["Variable"] = ["spikes_exc_ext_tot", "s_AMPA_ext_tot", "I_AMPA_ext_tot"]
        mean_field_ext_mean._data = mean_field_ext_mean._data.assign_coords(coords)
        # Plot main and internal synapses' time series:
        mean_field_ts[:, :"I_GABA"].plot_timeseries(plotter_config=plotter.config, 
                                                    per_variable=mean_field_ts.shape[1] > MAX_VARS_IN_COLS)
        # Plot time series of spiking_nodes_ids nodes' external synapses:
        for ind in netpyne_network_builder.spiking_nodes_inds:
            labels = ["spikes_exc_ext_%d" % ind, "s_AMPA_ext_%d" % ind, "I_AMPA_ext_%d" % ind]
            mean_field_ts[:, labels].plot_timeseries(plotter_config=plotter.config, 
                                                     per_variable=mean_field_ts.shape[1] > MAX_VARS_IN_COLS)
        # Plot total sum of external synapses' times series
        mean_field_ext_mean.plot_timeseries(plotter_config=plotter.config, 
                                            per_variable=mean_field_ts.shape[1] > MAX_VARS_IN_COLS)
        if mean_field_ts.number_of_labels > MIN_REGIONS_FOR_RASTER_PLOT:
            mean_field_ts[:, :"I_GABA"].plot_raster(plotter_config=plotter.config, 
                                                    per_variable=mean_field_ts.shape[1] > MAX_VARS_IN_COLS,
                                                    linestyle="--", alpha=0.5, linewidth=0.5)
            for ind in netpyne_network_builder.spiking_nodes_inds:
                labels = ["spikes_exc_ext_%d" % ind, "s_AMPA_ext_%d" % ind, "I_AMPA_ext_%d" % ind]
                mean_field_ts[:, labels].plot_raster(plotter_config=plotter.config, 
                                                     per_variable=mean_field_ts.shape[1] > MAX_VARS_IN_COLS,
                                                     linestyle="--", alpha=0.5, linewidth=0.5)
            mean_field_ext_mean.plot_raster(plotter_config=plotter.config, 
                                            per_variable=mean_field_ts.shape[1] > MAX_VARS_IN_COLS,
                                            linestyle="--", alpha=0.5, linewidth=0.5)
        del mean_field_ext_mean
                    
else:
    mean_field_ts = None

# Write results to file:
if mean_field_ts and writer:
    writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(
                                       mean_field_ts._data,
                                       connectivity=mean_field_ts.connectivity),
                           os.path.join(config.out.FOLDER_RES, mean_field_ts.title) + ".h5", 
                           recursive=False)

# Compute per neuron spikes' rates times series and plot them
if spikes_res and plot_per_neuron:
    from tvb.simulator.plot.base_plotter import pyplot
    spikeNet_analyzer.return_data = False
    rates_ts_per_neuron = \
        spikeNet_analyzer. \
            compute_spikeNet_rates_time_series(populations_devices=None, regions=None,
                                               computations_kwargs={}, data_kwargs={},
                                               return_spikes_trains=False, return_devices=False);
    if rates_ts_per_neuron is not None and rates_ts_per_neuron.size:
        # Regions in rows
        row = rates_ts_per_neuron.dims[2] if rates_ts_per_neuron.shape[2] > 1 else None
        if row is None:
            # Populations in rows
            row = rates_ts_per_neuron.dims[1] if rates_ts_per_neuron.shape[1] > 1 else None
            col = None
        else:
            # Populations in columns
            col = rates_ts_per_neuron.dims[1] if rates_ts_per_neuron.shape[1] > 1 else None
        pyplot.figure()
        rates_ts_per_neuron.plot(y=rates_ts_per_neuron.dims[3], row=row, col=col, cmap="jet")
        plotter.base._save_figure(figure_name="Spike rates per neuron")
        # del rates_ts_per_neuron # to free memory

### Plot per neuron SpikingNetwork time series

# Regions in rows
if spikeNet_ts is not None and spikeNet_ts.size:
    row = spikeNet_ts.dims[2] if spikeNet_ts.shape[2] > 1 else None
    if row is None:
        # Populations in rows
        row = spikeNet_ts.dims[3] if spikeNet_ts.shape[3] > 1 else None
        col = None
    else:
        # Populations in cols
         col = spikeNet_ts.dims[3] if spikeNet_ts.shape[3] > 1 else None
    for var in spikeNet_ts.coords[spikeNet_ts.dims[1]]:
        this_var_ts = spikeNet_ts.loc[:, var, :, :, :]
        this_var_ts.name = var.item()
        pyplot.figure()
        this_var_ts.plot(y=spikeNet_ts.dims[4], row=row, col=col, cmap="jet", figsize=FIGSIZE)
        plotter.base._save_figure(
            figure_name="Spiking Network variables' time series per neuron: %s" % this_var_ts.name)
    del spikeNet_ts # to free memory






