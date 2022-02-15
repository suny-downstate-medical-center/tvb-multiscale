# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_netpyne.config import Config, initialize_logger
from tvb_multiscale.tvb_netpyne.interfaces.models.builders.default import DefaultInterfaceBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder

from examples.example import default_example as default_example_base
from examples.models.red_wong_wang import red_wong_wang_excio_inhi_example as red_wong_wang_excio_inhi_example_base


def main_example(model_fun_to_run, netpyne_model_builder, tvb_netpyne_model_builder, **kwargs):

    some_args = {"populations_order": 100,
                 "spiking_proxy_inds": [60, 61], # superiortemporal L and R
                 "simulation_length": 220}
    kwargs.update(some_args)
    return model_fun_to_run(netpyne_model_builder, tvb_netpyne_model_builder,
                            config_type=Config, logger_initializer=initialize_logger, **kwargs)


def default_example(**kwargs):
    # otherwise: default_example_base
    return main_example(red_wong_wang_excio_inhi_example_base, DefaultExcIOInhIBuilder(), DefaultInterfaceBuilder, **kwargs)


if __name__ == "__main__":
    default_example()
