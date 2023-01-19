# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_netpyne.interfaces.models.wilson_cowan import WilsonCowanTVBNetpyneInterfaceBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.wilson_cowan import WilsonCowanBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.thalamic_VIM_exc_io_inh_i import WilsonCowanThalamicVIMBuilder

from examples.tvb_netpyne.example import main_example
from examples.models.wilson_cowan import wilson_cowan_example as wilson_cowan_example_base


def wilson_cowan_example(**kwargs):
    params = {
        "simulation_length": 500,
        # "model_params": {"lamda": 0.5}
    }
    netpyne_network_builder = WilsonCowanThalamicVIMBuilder()

    #change connectivity to Glasser
    kwargs['connectivity']='/home/docker/packages/tvb_data/tvb_data/connectivity/connectivity_Glasser.zip'
    kwargs.update(params)
    
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

    # NetPyNE model built already: including populations, connections, stimulation, recording

    # ??? TVB weight/scaling?

    # ??? TVB delay?

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
