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

    # netpyne_model_builder = WilsonCowanBuilder()
    netpyne_network_builder = WilsonCowanThalamicVIMBuilder()
    params["spiking_proxy_inds"] = [8, 8, 379, 371, 361] #M1R, PMC, brainstem, thalamusR, cerebellumL, muscle

    kwargs.update(params)
    
    # connectivity using glasser parcellation
    kwargs['connectivity'] = '/path/to/Glasser/connectivity'

    tvb_netpyne_interface_builder = WilsonCowanTVBNetpyneInterfaceBuilder()
    return main_example(wilson_cowan_example_base, netpyne_network_builder, tvb_netpyne_interface_builder, **kwargs)


if __name__ == "__main__":
    wilson_cowan_example()
