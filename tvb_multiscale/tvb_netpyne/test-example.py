import os
from collections import OrderedDict
import time
import numpy as np

# Importing and setting LIBRARY_PROFILE for TVB scripting
from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

# Importing and setting some configurations for tvb_multiscale and tvb_netpyne
from tvb_multiscale.tvb_netpyne.config import NetpyneConfig as Config

work_path = os.getcwd()
outputs_path = os.path.join(work_path, "netpyne-bound-outputs/RedWongWang")
config = Config(output_base=outputs_path)

config.figures.SHOW_FLAG = True 
config.figures.SAVE_FLAG = True
config.figures.FIG_FORMAT = 'png'
config.figures.DEFAULT_SIZE= config.figures.NOTEBOOK_SIZE
FIGSIZE = config.figures.DEFAULT_SIZE

# Importing and creating a Plotter class object
from tvb_multiscale.core.plot.plotter import Plotter
plotter = Plotter(config.figures)

# For interactive plotting:
# %matplotlib notebook  

# Otherwise:
# %matplotlib inline 

##################################################################################################################
# 1. Load structural data (minimally a TVB connectivity) & prepare TVB simulator (region mean field model, integrator, monitors etc)
##################################################################################################################

from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI


# # ----------------------1. Build a TVB simulator (model, integrator, monitors...)----------------------------------
# # Create a TVB simulator using the tvb-multiscale helper SimulatorBuilder class
# from tvb_multiscale.core.tvb.simulator_builder import SimulatorBuilder
# simulator_builder = SimulatorBuilder()
# simulator_builder.connectivity = config.DEFAULT_CONNECTIVITY_ZIP  
# simulator_builder.model = ReducedWongWangExcIOInhI
# model_params = {"G": np.array([20.0, ]), "J_i": np.array([1.0, ]), "lamda": np.array([0.0, ])}
# simulator = simulator_builder.build(**model_params)


# ----------------------------------------------------------------------------------------------------------------
# ---------------------------Uncomment below to create the TVB simulator manually:--------------------------------
# ----------------------------------------------------------------------------------------------------------------

# Configuring the connectivity:
from tvb.datatypes.connectivity import Connectivity
# Load default TVB connectivity from ZIP file:
connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
# Normalize connectivity weights
# Normalization with the maximum per-region/node sum of outgoing connections' weights
# i.e. normalisation_factor = numpy.max(numpy.abs(connectivity.weights.sum(axis=1))):
connectivity.weights = connectivity.scaled_weights(mode="region")
# Further normalization to scale and ceil the connectivity weights:
connectivity.weights /= np.percentile(connectivity.weights, 95)
connectivity.weights[connectivity.weights>1.0] = 1.0
connectivity.configure() # configure now in order to plot

from tvb.simulator.cosimulator import CoSimulator
# Create a TVB simulator and set all desired inputs
# (connectivity, model, surface, stimuli etc)
# We choose all defaults in this example
simulator = CoSimulator()
simulator.model = ReducedWongWangExcIOInhI()  # setting the node mean field model...
# ...and modifying some of its parameters:
simulator.model.G = np.array([20.0, ])     # Global cloupling scaling
simulator.model.J_i = np.array([1.0, ])    # Feedback inhibition
simulator.model.lamda = np.array([0.0, ])  # Feedforward inhibition

# Setting the connectivity:
simulator.connectivity = connectivity 

#Setting the coupling function:
from tvb.simulator.coupling import Linear
# TVB linear coupling of the form: a*x + b with default values of a = 1.0/256 and b = 0.0
simulator.coupling = Linear(a=np.array([1.0/256]), b=np.array([0.0]))

# Setting the integrator:
from tvb.simulator.integrators import HeunStochastic
simulator.integrator = HeunStochastic()
simulator.integrator.dt = 0.1 # the TVB time step of integration
simulator.integrator.noise.nsig = np.array([0.001]) # the amplitude (standard deviation) of noise 

# Setting a tuple of TVB monitors
from tvb.simulator.monitors import Raw  # , Bold, EEG
# Raw monitor records all state variables of the model
mon_raw = Raw(period=1.0)  # sampling period in ms
simulator.monitors = (mon_raw, )  # , Bold, EEG

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# TODO: Uncomment! Commented out for speed
# plotter.plot_tvb_connectivity(simulator.connectivity);

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
        

# Import and configure a NetpyneModelBuilder class:
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder

netpyne_model_builder = DefaultExcIOInhIBuilder(simulator, spiking_nodes_ids, config=config)

netpyne_model_builder.population_order = 100  # scalable population size

# Parameters for coupling between region nodes configured on the basis of the respective TVB parameters:
netpyne_model_builder.global_coupling_scaling = \
    netpyne_model_builder.tvb_simulator.coupling.a[0].item() * netpyne_model_builder.tvb_model.G[0].item()
netpyne_model_builder.lamda = netpyne_model_builder.tvb_model.lamda[0].item()


# # Using all default parameters for this example
# netpyne_model_builder.set_defaults()
# # The number of neurons for exc and inh populations:
# N_E = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_e)
# N_I = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_i)


# or...

from copy import deepcopy

# 1. Populations' configurations

population_neuron_model = "PYR"  # the NetPyNE spiking neuron model

# The number of neurons for exc and inh populations:
netpyne_model_builder.scale_e = 1.756  # scaling exc population size
netpyne_model_builder.scale_i = 0.43  # scaling inh population size

N_E = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_e)
N_I = int(netpyne_model_builder.population_order * netpyne_model_builder.scale_i)

# When any of the properties 
# model, params and scale
# depends on regions, 
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property
netpyne_model_builder.populations = [
    {"label": "E", # string label for population
     "model": population_neuron_model, # the NetPyNE spiking neuron model
     "nodes": None,  # list of regions' indices or None which means "all", 
                     # i.e., building this population on all spiking regions 
     "params": {}, # dict of parameters of the the NetPyNE spiking neuron model to be optionally modified
     "scale": netpyne_model_builder.scale_e},
    {"label": "I",
     "model": population_neuron_model,
     "nodes": None,
     "params": {},
     "scale": netpyne_model_builder.scale_i}
  ]

# 2. Within region-node connections
# When any of the properties model, conn_spec, weight, delay, receptor_type 
# depend on the regions,
# set a handle to a function with
# arguments (region_index=None) returning the corresponding property

exc_synapse_model = "exc" # NetPyNE synapse models 
inh_synapse_model = "inh"

# Dict of properties of connections among neurons:
conn_spec = {'rule': "all_to_all", 
             "allow_autapses": True, 'allow_multapses': True, 
             # parameters relating to other connectivity rules:
             "indegree": None, "outdegree": None, "N": None, "p": 0.1
            }

# from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import random_normal_weight, random_uniform_tvb_delay

# Default within region weights (sign determines exc or inh):
w_e = 1.0  # exc -> exc, exc -> inh
# w_e = random_normal_weight(w_e, sigma=0.1) 
# w_e = {"distribution": "normal", "mu": w_e, "sigma": 0.1*w_e}
w_ie = -netpyne_model_builder.tvb_model.J_i[0].item() # inh -> exc
# w_ie = random_normal_weight(w_ie, sigma=0.1) 
# w_ie =  {"distribution": "normal", "mu": w_ie, "sigma": 0.1 * np.abs(w_ie)}
w_ii = -1 # inh -> inh
# w_ii = random_normal_weight(w_ii, sigma=0.1) 
# w_ii =  {"distribution": "normal", "mu": w_ii, "sigma": 0.1 * np.abs(w_ii)}

within_node_delay = netpyne_model_builder.default_min_delay
#                     {"distribution": "uniform", 
#                      "low": np.minimum(netpyne_model_builder.default_min_delay, 
#                                        netpyne_model_builder.default_populations_connection["delay"]), 
#                      "high": np.maximum(netpyne_model_builder.tvb_dt, 
#                                         2 * netpyne_model_builder.default_populations_connection["delay"])}

netpyne_model_builder.populations_connections = [
     #              ->
    {"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
     "synapse_model": exc_synapse_model, "conn_spec": conn_spec,
     "weight": w_e, "delay": within_node_delay,
     "receptor_type": 0, # synaptic port to be modified for multisynapse NetPyNE spiking models
     "nodes": None},  # None means apply to all spiking regions
    {"source": "E", "target": "I",  # E -> I
     "synapse_model": exc_synapse_model, "conn_spec": conn_spec,
     "weight": w_e, "delay": within_node_delay,
     "receptor_type": 0, "nodes": None},  
    {"source": "I", "target": "E",  # I -> E
     "synapse_model": inh_synapse_model, "conn_spec": conn_spec, 
     "weight": w_ie, "delay": within_node_delay,
     "receptor_type": 0, "nodes": None},  #
    {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
     "synapse_model": inh_synapse_model, "conn_spec": conn_spec,
     "weight": w_ii, "delay": within_node_delay,
     "receptor_type": 0, "nodes": None}  
    ]

# 3. Among/Between region-node connections
        
from tvb_multiscale.core.spiking_models.builders.templates import scale_tvb_weight, tvb_delay
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import random_normal_tvb_weight, random_uniform_tvb_delay
    

tvb_weight_fun = lambda source_node, target_node: \
    scale_tvb_weight(source_node, target_node, 
                     netpyne_model_builder.tvb_weights, 
                     scale=netpyne_model_builder.global_coupling_scaling)
#     random_normal_tvb_weight(source_node, target_node, 
#                      netpyne_model_builder.tvb_weights, 
#                      scale=netpyne_model_builder.global_coupling_scaling, 
#                      sigma=0.1)
tvb_delay_fun = lambda source_node, target_node: \
                 tvb_delay(source_node, target_node, netpyne_model_builder.tvb_delays)
#                  random_uniform_tvb_delay(source_node, target_node, netpyne_model_builder.tvb_delays, 
#                                           low=netpyne_model_builder.tvb_dt, 
#                                           high=2 * netpyne_model_builder.tvb_dt, 
#                                           sigma=0.1)


# When any of the properties model, conn_spec, weight, delay, receptor_type 
# depends on regions, set a handle to a function with
# arguments (source_region_index=None, target_region_index=None)
netpyne_model_builder.nodes_connections = [
    #              ->
    {"source": "E", "target": ["E"],
     "synapse_model": exc_synapse_model, "conn_spec": conn_spec,
     "weight": tvb_weight_fun, "delay": tvb_delay_fun, "receptor_type": 0,
     "source_nodes": None, "target_nodes": None}  # None means apply to all
    ]

lamda = netpyne_model_builder.tvb_model.lamda[0]
print()
if lamda > 0: # if we want exc -> inh connections among regions:
    netpyne_model_builder.nodes_connections.append(
        {"source": "E", "target": ["I"],
         "synapse_model": exc_synapse_model, "conn_spec": conn_spec,
         "weight": lambda source_node, target_node: \
                        scale_tvb_weight(source_node, target_node, 
                                         netpyne_model_builder.tvb_weights, 
                                         scale=lamda * netpyne_model_builder.global_coupling_scaling), 
         "delay": tvb_delay_fun, "receptor_type": 0,
         "source_nodes": None, "target_nodes": None}  # None means apply to all
    )

# 4. Create a spike stimulus input device
# TODO: commented out temporarily for clearer debugging.
# netpyne_model_builder.input_devices = [
#     {"model": "spike_generator_placeholder", # the NetPyNE device model
#      # parameters of the device:
#      "params": {"rate": 7200.0, "origin": 0.0, "start": 0.1},  # "stop": 100.0
#      # connections between the devices and the target populations
#      #          stimulus label -> target populations labels
#      "connections": {"Stimulus": ["E", "I"]}, 
#      "nodes": None,         # None means apply to all spiking regions
#      "weights": 1.0, 
#      "delays": netpyne_model_builder.tvb_dt, 
# #      {"distribution": "uniform", 
# #       "low": netpyne_model_builder.tvb_dt, 
# #       "high": 2 * netpyne_model_builder.tvb_dt},
#      "receptor_type": 0}
#                                   ]  #
# 5. Creating  devices to be able to observe NetPyNE activity:

netpyne_model_builder.output_devices = []

# Spike recorders:
# TODO: commented out temporarily for clearer debugging.
# connections = OrderedDict({})
# #          label <- target population
# connections["E_spikes"] = "E"
# connections["I_spikes"] = "I"
# netpyne_model_builder.output_devices.append(
#     {"model": "spike_recorder", "params": {"record_to": "memory"}, # ascii
#      "connections": connections, "nodes": None})  # None means apply to all

# Labels have to be different

# TODO: uncomment when we have multimeter
# connections = OrderedDict({})
# #          label<- target population
# connections["E"] = "E"
# connections["I"] = "I"
# params = {"interval": 1.0, # ms
#           'record_from': ["V_m", "g_ex", "g_in"], "record_to": "memory"} 
# netpyne_model_builder.output_devices.append(
#     {"model": "multimeter", "params": params,
#      "connections": connections, "nodes": None})  # None means apply to all spiking regions
    

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# Now the NetpyneModelBuilder can configure and build the spiking network!
tvb_to_netpyne_state_variable = "R_e" # TODO: better name?
netpyne_model_builder.state_variable = tvb_to_netpyne_state_variable
netpyne_network = netpyne_model_builder.build_spiking_network()

##################################################################################################################
# 3. Build the TVB-NetPyNE interface
##################################################################################################################

# Build a TVB-NetPyNE interface with all the appropriate connections between the
# TVB and NetPyNE modelled regions

# Start by importing and generating a TVB-NetPyNE interface builder class:
from tvb_multiscale.tvb_netpyne.interfaces.builders.models.default import DefaultInterfaceBuilder 
tvb_netpyne_builder = \
    DefaultInterfaceBuilder(simulator, netpyne_network, spiking_nodes_ids, 
                            exclusive_nodes=True, populations_sizes=[N_E, N_I])

# Select the TVB -> NetPyNE interface mode...
tvb_to_netpyne_mode = "rate" # "rate", "current", "param"
# ... and whether NetPyNE should update TVB:
netpyne_to_tvb=True

# 1. Some basic configurations on conversions and scalings between the coarse and fine scale!:

tvb_netpyne_builder.w_tvb_to_current = 1000 * tvb_netpyne_builder.tvb_model.J_N[0]  # (nA of TVB -> pA of NetPyNE)
# WongWang model parameter r is in Hz, just like poisson_generator assumes in NetPyNE: # TODO: poisson generator?
tvb_netpyne_builder.w_tvb_to_spike_rate = 1.0
# We return from a NetPyNE spike_detector the ratio 
# number_of_population_spikes / number_of_population_neurons
# for every TVB time step, which is usually a quantity in the range [0.0, 1.0],
# as long as a neuron cannot fire twice during a TVB time step, i.e.,
# the TVB time step (usually 0.001 to 0.1 ms)
# is smaller than the neurons' refractory time, t_ref (usually 1-2 ms).
# For conversion to a rate, one has to do:
# w_spikes_to_tvb = 1/tvb_dt, to get it in spikes/ms, and
# w_spikes_to_tvb = 1000/tvb_dt, to get it in spikes/sec, i.e., Hz
# given WongWang model parameter r is in Hz but tvb dt is in ms:
tvb_netpyne_builder.w_spikes_to_tvb = 1000.0 / tvb_netpyne_builder.tvb_dt

# Some configurations for the connectivity from TVB to NetPyNE regions:
from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay, scale_tvb_weight
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates \
    import random_normal_tvb_weight, random_uniform_tvb_delay

# Parameters for the connectivity from TVB to NetPyNE regions, taken from the TVB model:
lamda = tvb_netpyne_builder.tvb_model.lamda[0].item()
G = tvb_netpyne_builder.tvb_model.G[0].item()
tvb_netpyne_builder.global_coupling_scaling = netpyne_model_builder.global_coupling_scaling


# TVB -> NetPyNE connectivity functions:

if tvb_to_netpyne_mode in ["rate", "current"]:
    
    tvb_weight_fun = lambda tvb_node_id, netpyne_node_id: \
                        scale_tvb_weight(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_weights, 
                                        scale=netpyne_model_builder.global_coupling_scaling)
#                         random_normal_tvb_weight(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_weights,
#                                                  scale=netpyne_model_builder.global_coupling_scaling,
#                                                  sigma=0.1)



    tvb_delay_fun = lambda tvb_node_id, netpyne_node_id: \
                        tvb_delay(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_delays)
#                         random_uniform_tvb_delay(tvb_node_id, netpyne_node_id, tvb_netpyne_builder.tvb_delays,
#                                                  low=tvb_netpyne_builder.tvb_dt, 
#                                                  high=2*tvb_netpyne_builder.tvb_dt,
#                                                  sigma=0.1)

#--------For spike transmission from TVB to NetPyNE devices acting as TVB proxy nodes with TVB delays:--------

# Mean spike rates are applied in parallel to all target neurons

if tvb_to_netpyne_mode == "rate":
    
    tvb_netpyne_builder.tvb_to_spikeNet_interfaces = [
        {"model": "spike_generator_placeholder",  # NetPyNE stimulation device to be used
         "params": {"allow_offgrid_times": False}, # parameters for the NetPyNE device
    # # ---------Properties potentially set as function handles with args (tvb_node_id=None)-------------------------
         "interface_weights": 1.0 * N_E, # Convert mean value to total value, i.e., scaling of TVB input
    # Applied outside NetPyNE for each interface device
    # -------Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)-----------
        "weights": tvb_weight_fun, "delays": tvb_delay_fun, "receptor_type": 0,
        # --------------------------------------------------------------------------------------------------------------
        #             TVB sv -> NetPyNE population
        "connections": {tvb_to_netpyne_state_variable: ["E"]},  # R_e - connection from TVB state variable to NetPyNE spiking population
        "source_nodes": None, "target_nodes": None}]  # None means all regions' connections 

    if lamda > 0.0:
        tvb_netpyne_builder.tvb_to_spikeNet_interfaces.append(
            {"model": "spike_generator_placeholder",
             "params": {"allow_offgrid_times": False},
        # # ---------Properties potentially set as function handles with args (tvb_node_id=None)-------------------------
             "interface_weights": lamda * N_E, # Convert mean value to total value
        # Applied outside NetPyNE for each interface device
        # -------Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)-----------
            "weights": tvb_weight_fun, "delays": tvb_delay_fun, "receptor_type": 0,
            # --------------------------------------------------------------------------------------------------------------
            #             TVB sv -> NetPyNE population
            "connections": {tvb_to_netpyne_state_variable: ["I"]}, # "R_e"
            "source_nodes": None, "target_nodes": None   # None means all regions' connections 
            } 
        )

# Mean currents are distributed to all target neurons

# TODO: to use coupling via currents, uncomment below and convert to netpyne
# if tvb_to_nest_mode == "current":

#     # --For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:--

#     tvb_nest_builder.tvb_to_spikeNet_interfaces = [
#         {"model": "dc_generator",  # NEST stimulation device to be used
#          "params": {}, # parameters for the NEST device
#     # Properties potentially set as function handles with args (tvb_node_id=None)
#     #   Applied outside NEST for each interface device
#          "interface_weights": 1.0,  # N_E / N_E, optional scaling of TVB input
#     # Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)
#          "weights": tvb_weight_fun, "delays": tvb_delay_fun,
#     #                   TVB sv -> NEST population
#          "connections": {"S_e": ["E"]}, # connection from TVB state variable to NEST spiking population
#          "source_nodes": None, "target_nodes": None}]  # None means all regions' connections 

#     if lamda > 0.0:
#         tvb_nest_builder.tvb_to_spikeNet_interfaces.append(
#            {"model": "dc_generator", "params": {},
#         # ---------Properties potentially set as function handles with args (tvb_node_id=None)
#         #   Applied outside NEST for each interface device
#              "interface_weights": lamda*N_E / N_I,
#         # Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)
#              "weights": tvb_weight_fun, "delays": tvb_delay_fun,
#         #                  TVB sv -> NEST population
#              "connections": {"S_e": ["I"]},
#              "source_nodes": None, "target_nodes": None}  # None means all regions' connections 
#         )


# Mean currents are distributed to all target neurons

# TODO: to use coupling via direct parameters update, uncomment below and convert to netpyne
# if tvb_to_nest_mode == "param":

#     # --For directly setting an external current parameter in NEST neurons instantaneously:--
#     tvb_nest_builder.tvb_to_spikeNet_interfaces = [
#         {"model": "current",  # The type of parameter
#          "parameter": "I_e",  # The name of NEST model parameter to target
#     # Properties potentially set as function handles with args (tvb_node_id=None)
#          "interface_weights": G,  # N_E / N_E, optional scaling of TVB input
#     #                  TVB sv -> NEST population
#          "connections": {"S_e": ["E"]}, # connection from TVB state variable to NEST spiking population
#          "nodes": None}]  ## None means all spiking regions
#     if lamda > 0.0:
#         # Coupling to inhibitory populations as well (feedforward inhibition):
#         tvb_nest_builder.tvb_to_spikeNet_interfaces.append(
#         {
#             "model": "current", "parameter": "I_e",
#     # Properties potentially set as function handles with args (tvb_node_id=None)
#             "interface_weights": lamda * G * N_E / N_I,
#     #                     TVB sv -> NEST population
#             "connections": {"S_e": ["I"]},
#             "nodes": None}  # None means all spiking regions
#     )

# NetPyNE -> TVB update:
if netpyne_to_tvb:
    connections = OrderedDict() # connection from NetPyNE spiking population to TVB state variable
    #            TVB sv <- NetPyNE
    connections["Rin_e"] = ["E"]
    connections["Rin_i"] = ["I"]
    tvb_netpyne_builder.spikeNet_to_tvb_interfaces = [
        {"model": "spike_recorder",  # NetPyNE device to record spikes
         "params": {},  # Optional parameters to the NetPyNE device 
    # ----Properties potentially set as function handles with args (netpyne_node_id=None)--
         "interface_weights": 1.0, # optional scaling of NetPyNE -> TVB update
         "delays": 0.0, # optional delay of NetPyNE -> TVB update
    # ---------------------------------------------------------------------------------
         "connections": connections, "nodes": None}]  # None means all spiking regions here

# Finally build the TVB<->NetPyNE interface model:
tvb_netpyne_model = tvb_netpyne_builder.build_interface(tvb_to_netpyne_mode=tvb_to_netpyne_mode, netpyne_to_tvb=netpyne_to_tvb)

# Print a summary report of the TVB<->NetPyNE interface model and, optionally, its connectivity:
print(tvb_netpyne_model.print_str(detailed_output=True, connectivity=False))  # True, will take some time!

##################################################################################################################
# 4. Configure simulator, simulate, gather results
##################################################################################################################

# Configure the simulator including the TVB-NetPyNE interface model...
simulator.configure(tvb_spikeNet_interface=tvb_netpyne_model)
# ...and simulate!
simulation_length=110.0  # Set at least 1100.0 for a meaningful simulation
transient = simulation_length/11  # a transient to be removed for rate and correlation computations
t = time.time()
results = simulator.run(simulation_length=simulation_length)

# TODO: is below ever needed?
# Integrate NetPyNE one more NetPyNE time step so that multimeters get the last time point
# unless you plan to continue simulation later
# simulator.run_spiking_simulator(0.05)

print("\nSimulated in %f secs!" % (time.time() - t))

# Clean-up NetPyNE simulation
# simulator.tvb_spikeNet_interface.nest_instance.Cleanup()

##################################################################################################################
# 5. Plot results and write them to HDF5 files
##################################################################################################################

# Set plot_per_neuron to False for faster plotting of only mean field variables and dates:
plot_per_neuron = True 
MAX_VARS_IN_COLS = 3
MAX_REGIONS_IN_ROWS = 10
MIN_REGIONS_FOR_RASTER_PLOT = 9
# from examples.plot_write_results import plot_write_results
# populations = []
# populations_sizes = []
# for pop in nest_model_builder.populations:
#     populations.append(pop["label"])
#     populations_sizes.append(int(np.round(pop["scale"] * nest_model_builder.population_order)))
# plot_write_results(results, simulator, populations=populations, populations_sizes=populations_sizes, 
#                    transient=transient, tvb_state_variable_type_label="State Variables", 
#                    tvb_state_variables_labels=simulator.model.variables_of_interest, 
#                    plot_per_neuron=plot_per_neuron, plotter=plotter, config=config)

# If you want to see what the function above does, take the steps, one by one
try:
    # We need framework_tvb for writing and reading from HDF5 files
    from tvb_multiscale.core.io.h5_writer import H5Writer
    writer = H5Writer()
except:
    writer = False
    
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

# Put the results in a Timeseries instance
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion

source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
        data=results[0][1], time=results[0][0],
        connectivity=simulator.connectivity,
        labels_ordering=["Time", "State Variable", "Region", "Neurons"],
        labels_dimensions={"State Variable": list(simulator.model.variables_of_interest),
                           "Region": simulator.connectivity.region_labels.tolist()},
        sample_period=simulator.integrator.dt)
source_ts.configure()

t = source_ts.time
    
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
n_spiking_nodes = len(simulator.tvb_spikeNet_interface.spiking_nodes_ids)
source_ts_netpyne = source_ts[:, :, simulator.tvb_spikeNet_interface.spiking_nodes_ids]
print(f"source_ts_netpyne: {source_ts_netpyne}")
print(f"plotter config {plotter.config}")
source_ts_netpyne.plot_timeseries(plotter_config=plotter.config, 
                               hue="Region" if source_ts_netpyne.shape[2] > MAX_REGIONS_IN_ROWS else None, 
                               per_variable=source_ts_netpyne.shape[1] > MAX_VARS_IN_COLS, 
                               figsize=FIGSIZE, figname="Spiking nodes TVB Time Series");






