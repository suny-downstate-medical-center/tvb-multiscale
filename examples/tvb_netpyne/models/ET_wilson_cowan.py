# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_netpyne.interfaces.models.wilson_cowan import WilsonCowanTVBNetpyneInterfaceBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.wilson_cowan import WilsonCowanBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.thalamic_VIM_exc_io_inh_i import WilsonCowanThalamicVIMBuilder

from examples.tvb_netpyne.example import main_example
from examples.models.wilson_cowan import wilson_cowan_example as wilson_cowan_example_base
# 0------
from IPython.display import Image, display
import os
from collections import OrderedDict
import time
import numpy as np
from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)
# 1  ------
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.monitors import Raw  # , Bold, EEG

from tvb_multiscale.tvb_netpyne.config import *

def wilson_cowan_example(**kwargs):
    params = {
        "simulation_length": 500,
        # "model_params": {"lamda": 0.5}
    }

    #(0) Setup stuff
    # For a minimal example, select:
    N_REGIONS = None # total TVB brain regions
    NETPYNE_NODES_INDS = np.array([0, 1])  # the brain region nodes to place spiking networks from [0, N_REGIONS-1] interval
    N_NEURONS = 100 # number of neurons per spiking population

    # Interface basic configurations:
    INTERFACE_MODEL = "RATE"  # The only available option for NetPyNE so far
    INTERFACE_COUPLING_MODE = "spikeNet"  # "spikeNet" # "TVB"
    # -----------------------------------------------
    work_path = os.getcwd()
    outputs_path = os.path.join(work_path, "outputs/NetPyNE_wilsoncowan_%s_%s" % (INTERFACE_MODEL, INTERFACE_COUPLING_MODE))

    config = Config(output_base=outputs_path)
    config.figures.SHOW_FLAG = True 
    config.figures.SAVE_FLAG = True
    config.figures.FIG_FORMAT = 'png'
    config.figures.DEFAULT_SIZE= config.figures.NOTEBOOK_SIZE
    FIGSIZE = config.figures.DEFAULT_SIZE

    from tvb_multiscale.core.plot.plotter import Plotter
    plotter = Plotter(config.figures)

    #1. Load structural data (minimally a TVB connectivity) & prepare TVB simulator (region mean field model, integrator, monitors etc)
    # This would run on TVB only before creating any multiscale cosimulation interface connections.
    from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI

    # Load full TVB connectome connectivity: Glasser parcellation
    data_path = os.path.expanduser("~/packages/tvb-multiscale/examples/tvb_netpyne/models")
    conn_path = os.path.join(data_path, "temp_connectivity_Glasser")

    w = np.loadtxt(os.path.join(conn_path, "weights.txt")) #np.zeros((11,11))
    c = np.loadtxt(os.path.join(conn_path, "centres.txt"), usecols=range(1,3))
    rl = np.loadtxt(os.path.join(conn_path, "region_labels.txt"), dtype="str", usecols = 1)
    t = np.loadtxt(os.path.join(conn_path, "tract_lengths.txt"))
    
    ######## PETERSEN ATLAS ############
    #doesn't do anything for now, except change variable name used from this point onwards
    # later, if we add additional connections from e.g. another atlas, we can add/modify/merge the values/labels and specify the new filenames here
    wTVB = w 
    cTVB = c
    rlTVB = rl
    tTVB = t
    #####################################

    number_of_regions = len(rlTVB)
    speed = np.array([4.0])

    connTVB = Connectivity(region_labels=rlTVB, weights=wTVB, centres=cTVB, tract_lengths=tTVB, speed=speed)

    # -------------- Pick a minimal brain of only the first N_REGIONS regions: ----------------
    if N_REGIONS is not None:
        connTVB.number_of_regions = N_REGIONS
        connTVB.region_labels = connTVB.region_labels[:N_REGIONS]
        connTVB.centres = connTVB.centres[:N_REGIONS]
        connTVB.areas = connTVB.areas[:N_REGIONS]
        connTVB.orientations = connTVB.orientations[:N_REGIONS]
        connTVB.hemispheres = connTVB.hemispheres[:N_REGIONS]
        connTVB.cortical = connTVB.cortical[:N_REGIONS]
        connTVB.weights = connTVB.weights[:N_REGIONS][:, :N_REGIONS]
        connTVB.tract_lengths = connTVB.tract_lengths[:N_REGIONS][:, :N_REGIONS]
    # -----------------------------------------------------------------------------------------

    # Remove diagonal self-connections:
    np.fill_diagonal(connTVB.weights, 0.0)

    # Normalize connectivity weights
    connTVB.weights = connTVB.scaled_weights(mode="region")
    connTVB.weights /= np.percentile(connTVB.weights, 99)

    # CoSimulator builder--------------------------------
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorSerialBuilder

    simulator_builder = CoSimulatorSerialBuilder()
    simulator_builder.config = config
    simulator_builder.model = ThalamicVIMBuilder()
    simulator_builder.connectivity = connTVB
    simulator_builder.model_params = model_params
    simulator_builder.initial_conditions = np.array([0.0])

    simulator_builder.configure()
    simulator_builder.print_summary_info_details(recursive=1)

    simulator = simulator_builder.build()
    # -----

    simulator.configure()


    simulator.print_summary_info_details(recursive=1)

    # Plot TVB connectome:
    plotter.plot_tvb_connectivity(simulator.connectivity)


    #(2) NetPyNE Network Builder
    netpyne_network_builder = WilsonCowanThalamicVIMBuilder()

    #change connectivity to Glasser
    kwargs['connectivity']='/home/docker/packages/tvb_data/tvb_data/connectivity/connectivity_Glasser.zip'
    kwargs.update(params)
    
    netpyne_network_builder.tvb_to_spiking_dt_ratio = 2 # 2 NetPyNE integration steps for 1 TVB integration step
    netpyne_network_builder.monitor_period = 1.0

    #TVB (Glasser parcellation) regions to be substituted with spiking populations (netpyne pops)
    #M1R (8), PMC (8), brainstem (379), thalamusR (371), cerebellumL (361), muscle (NA)   ###DEFAULT[50, 58,] #50:precentral_R, 58:superiofrontal_R (taken as temp proxy for premotor cortex)

    netpyne_network_builder.primary_motor_cortex_R = 8
    netpyne_network_builder.cerebellar_cortex_L = 361
    netpyne_network_builder.thalamus_R = 371
    netpyne_network_builder.brainstem = 379

    params["spiking_proxy_inds"] = [
        netpyne_network_builder.primary_motor_cortex_R,
        netpyne_network_builder.cerebellar_cortex_L,
        netpyne_network_builder.thalamus_R,
        netpyne_network_builder.brainstem,
    ]
    kwargs.update(params)

    tvb_netpyne_interface_builder = WilsonCowanTVBNetpyneInterfaceBuilder()

    # input/output interfaces
    import numpy as np
    tvb_netpyne_interface_builder.default_coupling_mode == "spikeNet"
    if tvb_netpyne_interface_builder.default_coupling_mode == "spikeNet":
        pass
        # coupling is computing in NetPyNE, we need as many TVB proxies 
        # as TVB regions coupling to STN and Striatum
        #??? proxy_inds = np.unique(np.concatenate([CTXtoSTNinds, CTXtoSNinds]))    


    return main_example(wilson_cowan_example_base, netpyne_network_builder, tvb_netpyne_interface_builder, **kwargs)


if __name__ == "__main__":
    wilson_cowan_example()
