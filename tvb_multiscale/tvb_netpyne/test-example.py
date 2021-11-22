import os

from collections import OrderedDict
import time
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_netpyne.config import *

work_path = os.getcwd()
data_path = os.path.join(work_path.split("tvb_netpyne")[0], "data")
outputs_path = os.path.join(work_path, "netpyne-outputs/RedWongWang")
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
simulator.model.lamda = np.array([0.0, ]) # Feedforward inhibition
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
    if simulator.connectivity.region_labels[id].find("hippo") > 0:
        spiking_nodes_ids.append(id)

# originally - WWDeco2014Builder        
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder

# Build a NetPyNE network model with the corresponding builder
netpyne_model_builder = DefaultExcIOInhIBuilder(simulator, spiking_nodes_ids, config=config)
netpyne_model_builder.configure()

N_E = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_e)
N_I = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_i)


# # Using all default parameters for this example
netpyne_model_builder.set_defaults()

# or...

# ----------------------------------------------------------------------------------------------------------------
# ----Uncomment below to modify the builder by changing the default options:--------------------------------------
# ----------------------------------------------------------------------------------------------------------------
from copy import deepcopy

population_neuron_model = "PYR"  # the NetPyNE spiking neuron model # TODO: use same is internally

netpyne_model_builder.population_order = 100

netpyne_model_builder.scale_e = 1.6
netpyne_model_builder.scale_i = 0.4

N_E = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_e)
N_I = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_i)

netpyne_model_builder.global_coupling_scaling = \
    netpyne_model_builder.tvb_serial_sim["coupling.a"][0].item() * \
    netpyne_model_builder.tvb_serial_sim["model.G"][0].item()
netpyne_model_builder.lamda = netpyne_model_builder.tvb_serial_sim["model.lamda"][0].item()


# When any of the properties model, params and scale below depends on regions,
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property

def param_fun(node_index, params, weight):
    w_E_ext = \
        weight * netpyne_model_builder.tvb_weights[:, node_index]
    w_E_ext[node_index] = 1.0  # this is external input weight to this node
    out_params = deepcopy(params)
    out_params.update({"w_E_ext": w_E_ext})
    return out_params
    

common_params = {
    "V_th": -50.0, "V_reset": -55.0, "E_L": -70.0, "E_ex": 0.0, "E_in": -70.0,                       # mV
    "tau_decay_AMPA": 2.0, "tau_decay_GABA_A": 10.0, "tau_decay_NMDA": 100.0, "tau_rise_NMDA": 2.0,  # ms
    "s_AMPA_ext_max": N_E * np.ones((netpyne_model_builder.number_of_regions,)).astype("f"), 
    "N_E": N_E, "N_I": N_I, "epsilon": 1.0  # /N_E
}
params_E = {
    "C_m": 500.0,    # pF
    "g_L": 25.0,     # nS
    "t_ref": 2.0,    # ms
    "g_AMPA_ext": 3.37, "g_AMPA": 0.065, "g_NMDA": 0.20, "g_GABA_A": 10.94,  # nS
    "w_E": netpyne_model_builder.tvb_serial_sim["model.w_p"][0].item(), 
    "w_I": netpyne_model_builder.tvb_serial_sim["model.J_i"][0].item()
}
params_E.update(common_params)
netpyne_model_builder.params_E = \
    lambda node_index: param_fun(node_index, params_E,
                                 weight=netpyne_model_builder.global_coupling_scaling)

params_I = {
    "C_m": 200.0,  # pF
    "g_L": 20.0,   # nS
    "t_ref": 1.0,  # ms
    "g_AMPA_ext": 2.59, "g_AMPA": 0.051,"g_NMDA": 0.16, "g_GABA_A": 8.51,  # nS
    "w_E": 1.0, "w_I": 1.0
}
params_I.update(common_params)
netpyne_model_builder.params_I = \
    lambda node_index: param_fun(node_index, params_I,
                                 weight=netpyne_model_builder.lamda * netpyne_model_builder.global_coupling_scaling)

# Populations' configurations
# When any of the properties model, params and scale below depends on regions,
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property
netpyne_model_builder.populations = [
    {"label": "E", "model": population_neuron_model,
     "nodes": None,  # None means "all"
     "params": netpyne_model_builder.params_E,
     "scale": netpyne_model_builder.scale_e},
    {"label": "I", "model": population_neuron_model,
     "nodes": None,  # None means "all"
     "params": netpyne_model_builder.params_I,
     "scale": netpyne_model_builder.scale_i}
  ]

# Within region-node connections
# When any of the properties model, conn_spec, weight, delay, receptor_type below
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property
# TODO: be mindful that for NetPyNE it's done different way (weight 1.0 and -1.0 for E and I respectively, and same 'static_synapse')
synapse_model_E = "exc"
synapse_model_I = "inh"
conn_spec = {"allow_autapses": True, 'allow_multapses': True, 'rule': "all_to_all",
             "indegree": None, "outdegree": None, "N": None, "p": 0.1}

w_E = 0.1 # {"distribution": "normal", "mu": 1.0, "sigma": 0.1}
w_I = 0.1 # {"distribution": "normal", "mu": -1.0, "sigma": 0.1}

within_node_delay = netpyne_model_builder.default_min_delay
#                     {"distribution": "uniform", 
#                      "low": np.minimum(netpyne_model_builder.default_min_delay, 
#                                        netpyne_model_builder.default_populations_connection["delay"]), 
#                      "high": np.maximum(netpyne_model_builder.tvb_dt, 
#                                         2*netpyne_model_builder.default_populations_connection["delay"])}

netpyne_model_builder.populations_connections = [
     #              ->
    {"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
     "synapse_model": synapse_model_E, "conn_spec": conn_spec,
     "weight": w_E, "delay": within_node_delay,
     "receptor_type": 0, "nodes": None},  # None means apply to all
    {"source": "E", "target": "I",  # E -> I
     "synapse_model": synapse_model_E, "conn_spec": conn_spec,
     "weight": w_E, "delay": within_node_delay,
     "receptor_type": 0, "nodes": None},  # None means apply to all
    {"source": "I", "target": "E",  # I -> E
     "synapse_model": synapse_model_I, "conn_spec": conn_spec, 
     "weight": w_I, "delay": within_node_delay,
     "receptor_type": 0, "nodes": None},  # None means apply to all
    {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
     "synapse_model": synapse_model_I, "conn_spec": conn_spec,
     "weight": w_I, "delay": within_node_delay,
     "receptor_type": 0, "nodes": None}  # None means apply to all
    ]


# Among/Between region-node connections
# Given that only the AMPA population of one region-node couples to
# all populations of another region-node,
# we need only one connection type
        
# When any of the properties model, conn_spec, weight, delay, receptor_type below
# depends on regions, set a handle to a function with
# arguments (source_region_index=None, target_region_index=None)

from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import random_uniform_tvb_delay
    
lamda = netpyne_model_builder.tvb_serial_sim["model.lamda"][0]

tvb_delay_fun = lambda source_node, target_node: \
                 tvb_delay(source_node, target_node, netpyne_model_builder.tvb_delays)
#                  random_uniform_tvb_delay(source_node, target_node, netpyne_model_builder.tvb_delays, 
#                                           low=netpyne_model_builder.tvb_dt, 
#                                           high=2*netpyne_model_builder.tvb_dt, 
#                                           sigma=0.1)

receptor_by_source_region = lambda source_node, target_node: int(source_node + 1)

# Total excitatory spikes of one region node will be distributed to
netpyne_model_builder.nodes_connections = [
    #              ->
    {"source": "E", "target": ["E"],
     "synapse_model": synapse_model_E, "conn_spec": conn_spec,
     "weight": w_E, "delay": tvb_delay_fun,  
    # Each region emits spikes in its own port:
     "receptor_type": receptor_by_source_region,
     "source_nodes": None, "target_nodes": None}  # None means apply to all
    ]

if lamda > 0:
    netpyne_model_builder.nodes_connections.append(
        {"source": "E", "target": ["I"],
         "synapse_model": synapse_model_E, "conn_spec": conn_spec,
         "weight": w_E, "delay": tvb_delay_fun,  
        # Each region emits spikes in its own port:
         "receptor_type": receptor_by_source_region,
         "source_nodes": None, "target_nodes": None}  # None means apply to all
    )
    
    
# Creating  devices to be able to observe NetPyNE activity:

netpyne_model_builder.output_devices = []

# TODO: commented out temporarily for clearer debugging.
# connections = OrderedDict({})
# #          label <- target population
# connections["E_spikes"] = "E"
# connections["I_spikes"] = "I"
# netpyne_model_builder.output_devices.append(
#     {"model": "spike_recorder", "params": {"record_to": "memory"},
#      "connections": connections, "nodes": None})  # None means apply to all

# Labels have to be different

# TODO: uncomment when we have multimeter
# connections = OrderedDict({})
# #               label    <- target population
# connections["E"] = "E"
# connections["I"] = "I"
# record_from = ["V_m", "I_L", "I_e",
#                "spikes_exc", "s_AMPA", "I_AMPA",
#                "x_NMDA", "s_NMDA", "I_NMDA",
#                "spikes_inh", "s_GABA", "I_GABA"]
# for i_node in range(netpyne_model_builder.number_of_regions):
#     record_from.append("spikes_exc_ext_%d" % i_node)
#     record_from.append("s_AMPA_ext_%d" % i_node)
#     record_from.append("I_AMPA_ext_%d" % i_node)
# params = {"interval": 1.0, 'record_from': record_from, "record_to": "memory"}  # ms
# netpyne_model_builder.output_devices.append(
#     {"model": "multimeter", "params": params,
#      "connections": connections, "nodes": None})  # None means apply to all
    
# Create a spike stimulus input device
# TODO: commented out temporarily for clearer debugging.
# netpyne_model_builder.input_devices = [
#     {"model": "poisson_generator",
#      "params": {"rate": 2400.0, "origin": 0.0, "start": 0.1},  # "stop": 100.0
#      "connections": {"Stimulus": ["E", "I"]}, 
#      "nodes": None,         # None means apply to all
#      "weights": w_E, 
#      "delays": netpyne_model_builder.tvb_dt, 
# #      {"distribution": "uniform", 
# #                 "low": netpyne_model_builder.tvb_dt, 
# #                 "high": 2*netpyne_model_builder.tvb_dt},
#     "receptor_type": lambda target_node: int(target_node + 1)},
#                                   ]  #

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

tvb_to_netpyne_state_variable = "R_e" # TODO: better name?
netpyne_model_builder.state_variable = tvb_to_netpyne_state_variable
netpyne_network = netpyne_model_builder.build(set_defaults=False) # or true, if netpyne_model_builder.set_default() is not yet ran

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

tvb_to_netpyne_mode = "rate"  # "rate", "current", "param"
netpyne_to_tvb = True

# Using all default parameters for this example

# or...


# ----------------------------------------------------------------------------------------------------------------
# ----Uncomment below to modify the builder by changing the default options:--------------------------------------
# ----------------------------------------------------------------------------------------------------------------

from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay, scale_tvb_weight
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates \
    import random_normal_tvb_weight, random_uniform_tvb_delay, receptor_by_source_region

lamda = netpyne_model_builder.tvb_serial_sim["model.lamda"][0].item()
G = netpyne_model_builder.tvb_serial_sim["model.G"][0].item()
tvb_netpyne_builder.global_coupling_scaling = G * netpyne_model_builder.tvb_serial_sim["coupling.a"][0].item()


# TVB -> NetPyNE

if tvb_to_netpyne_mode in ["rate", "current"]:

    tvb_delay_fun = lambda tvb_node_id, netpyne_node_id: \
                        tvb_delay(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_delays)
#                         random_uniform_tvb_delay(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_delays,
#                                                  low=tvb_netpyne_builder.tvb_dt, 
#                                                  high=2*tvb_netpyne_builder.tvb_dt,
#                                                  sigma=0.1)

# --------For spike transmission from TVB to NetPyNE devices acting as TVB proxy nodes with TVB delays:--------

# Mean spike rates are applied in parallel to all target neurons

if tvb_to_netpyne_mode == "rate":
    weights = 1.0 # {"distribution": "normal", "mu": 1.0, "sigma": 0.1}
    receptor_by_source_region = lambda source_node, target_node: int(source_node + 1)
    
    tvb_netpyne_builder.tvb_to_spikeNet_interfaces = [
        {"model": "spike_generator_placeholder",
         "params": {"allow_offgrid_times": False},
    # # ---------Properties potentially set as function handles with args (tvb_node_id=None)-------------------------
         "interface_weights": 1.0 * N_E, # Convert mean value to total value
    # Applied outside NetPyNE for each interface device
    # -------Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)-----------
        "weights": weights, "delays": tvb_delay_fun,
        "receptor_type": receptor_by_source_region,
        # --------------------------------------------------------------------------------------------------------------
        #             TVB sv -> NetPyNE population
        "connections": {tvb_to_netpyne_state_variable: ["E"]},
        "source_nodes": None, "target_nodes": None}]  # None means all here

    if lamda > 0.0:
        tvb_netpyne_builder.tvb_to_spikeNet_interfaces.append(
            {"model": "spike_generator_placeholder",
             "params": {"allow_offgrid_times": False},
        # # ---------Properties potentially set as function handles with args (tvb_node_id=None)-------------------------
             "interface_weights": 1.0 * N_E, # Convert mean value to total value
        # Applied outside NetPyNE for each interface device
        # -------Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)-----------
            "weights": weights, "delays": tvb_delay_fun,
            "receptor_type": receptor_by_source_region,
            # --------------------------------------------------------------------------------------------------------------
            #             TVB sv -> NetPyNE population
            "connections": {tvb_to_netpyne_state_variable: ["I"]},
            "source_nodes": None, "target_nodes": None
            }
        )


    
# Mean currents are distributed to all target neurons

# TODO: to use coupling via currents, uncomment below and convert to netpyne
# if tvb_to_netpyne_mode == "current":
    
#     tvb_weight_fun = \
#         lambda tvb_node_id, netpyne_node_id: \
#             scale_tvb_weight(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_weights, 
#                              scale=tvb_netpyne_builder.global_coupling_scaling)
# #             random_normal_tvb_weight(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_weights, 
# #                                      scale=tvb_netpyne_builder.global_coupling_scaling, sigma=0.1)

#     # --------For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:--------

#     tvb_netpyne_builder.tvb_to_spikeNet_interfaces = [
#         {"model": "dc_generator", "params": {},
#     # Properties potentially set as function handles with args (tvb_node_id=None)
#     #   Applied outside NEST for each interface device
#          "interface_weights": 17.5,  # N_E / N_E
#     # Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)
#          "weights": tvb_weight_fun,
#          "delays": tvb_delay_fun,
#     #                   TVB sv -> NEST population
#          "connections": {"S_e": ["E"]},
#          "source_nodes": None, "target_nodes": None}]  # None means all here

#     if lamda > 0.0:
#         tvb_netpyne_builder.tvb_to_spikeNet_interfaces.append(
#            {"model": "dc_generator", "params": {},
#         # ---------Properties potentially set as function handles with args (tvb_node_id=None)
#         #   Applied outside NEST for each interface device
#              "interface_weights": lamda * N_E / N_I,
#         # Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)
#              "weights": tvb_weight_fun,
#              "delays": tvb_delay_fun,
#         #                  TVB sv -> NEST population
#              "connections": {"S_e": ["I"]},
#              "source_nodes": None, "target_nodes": None}
#         )

# TODO: to use coupling via direct parameters update, uncomment below and convert to netpyne
# if tvb_to_netpyne_mode == "param":

#     # --------For directly setting an external current parameter in NEST neurons instantaneously:--------
#     tvb_netpyne_builder.tvb_to_spikeNet_interfaces = [
#         {"model": "current",  "parameter": "I_e",
#     # Properties potentially set as function handles with args (tvb_node_id=None)
#          "interface_weights": G,  # N_E / N_E
#     #                  TVB sv -> NEST population
#          "connections": {"S_e": ["E"]},
#          "nodes": None}]  # None means all here
#     if lamda > 0.0:
#         # Coupling to inhibitory populations as well (feedforward inhibition):
#         tvb_netpyne_builder.tvb_to_spikeNet_interfaces.append(
#         {
#             "model": "current", "parameter": "I_e",
#     # Properties potentially set as function handles with args (tvb_node_id=None)
#             "interface_weights": lamda * G * N_E / N_I,
#     #                     TVB sv -> NEST population
#             "connections": {"S_e": ["I"]},
#             "nodes": None}
#     )

    

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

    
tvb_netpyne_builder.w_tvb_to_current = 1000 * netpyne_model_builder.tvb_serial_sim["model.J_N"][0]  # (nA of TVB -> pA of NetPyNE)
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






