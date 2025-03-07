# -*- coding: utf-8 -*-

from logging import Logger
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.interfaces.tvb.builders import TVBSpikeNetInterfaceBuilder
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels, SpikeNetToTVBModels
from tvb_multiscale.core.interfaces.spikeNet.builders import SpikeNetProxyNodesBuilder, SpikeNetInterfaceBuilder, \
    SpikeNetRemoteInterfaceBuilder, SpikeNetTransformerInterfaceBuilder,  \
    SpikeNetOutputTransformerInterfaceBuilder, SpikeNetInputTransformerInterfaceBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices

from tvb_multiscale.tvb_annarchy.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.interfaces.interfaces import \
    ANNarchyOutputInterface, ANNarchyInputInterface, \
    ANNarchySenderInterface, ANNarchyReceiverInterface, \
    ANNarchyTransformerSenderInterface, ANNarchyReceiverTransformerInterface, \
    TVBtoANNarchyInterface, ANNarchyToTVBInterface
from tvb_multiscale.tvb_annarchy.interfaces.io import \
    ANNarchySpikeMonitorSet, ANNarchySpikeMonitorTotalSet, \
    ANNarchyMonitorSet, ANNarchyMonitorMeanSet, ANNarchyMonitorTotalSet, \
    ANNarchyTimedPoissonPopulationSet, ANNarchySpikeSourceArraySet, ANNarchyTimedArraySet, \
    ANNarchyHomogeneousCorrelatedSpikeTrainsSet
from tvb_multiscale.tvb_annarchy.annarchy_models.network import ANNarchyNetwork
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import create_device, connect_device


TVBtoANNarchyModels = TVBtoSpikeNetModels
ANNarchyToTVBModels = SpikeNetToTVBModels


class ANNarchyInputProxyModels(Enum):
    RATE = ANNarchyTimedPoissonPopulationSet
    RATE_TO_SPIKES = ANNarchyTimedPoissonPopulationSet
    RATE_TO_CORRELATED_SPIKES = ANNarchyHomogeneousCorrelatedSpikeTrainsSet
    SPIKES = ANNarchySpikeSourceArraySet
    CURRENT = ANNarchyTimedArraySet


class ANNarchyOutputProxyModels(Enum):
    SPIKES = ANNarchySpikeMonitorSet
    SPIKES_MEAN = ANNarchySpikeMonitorTotalSet
    POTENTIAL = ANNarchyMonitorSet
    POTENTIAL_MEAN = ANNarchyMonitorMeanSet
    POTENTIAL_TOTAL = ANNarchyMonitorTotalSet


class DefaultTVBtoANNarchyModels(Enum):
    RATE = ANNarchyInputProxyModels.RATE.name
    SPIKES = ANNarchyInputProxyModels.SPIKES.name
    CURRENT =ANNarchyInputProxyModels.CURRENT.name


class DefaultANNarchytoTVBModels(Enum):
    SPIKES = ANNarchyOutputProxyModels.SPIKES_MEAN.name
    POTENTIAL = ANNarchyOutputProxyModels.POTENTIAL_MEAN.name


class ANNarchyProxyNodesBuilder(SpikeNetProxyNodesBuilder):

    """ANNarchyProxyNodesBuilder class"""

    spiking_network = Attr(label="ANNarchy Network",
                           doc="""The instance of ANNarchyNetwork class""",
                           field_type=ANNarchyNetwork,
                           required=True)


    @property
    def annarchy_network(self):
        return self.spiking_network

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance

    @property
    def spikeNet_min_delay(self):
        return self.annarchy_instance.dt()

    @property
    def annarchy_min_delay(self):
        return self.annarchy_instance.dt()

    def _build_and_connect_devices(self, interface, **kwargs):
        return build_and_connect_devices(interface, create_device, connect_device,
                                         self.spiking_network.brain_regions,
                                         self.config, annarchy_instance=self.annarchy_instance,
                                         input_proxies=self.spiking_network.input_proxies, **kwargs)

    def _build_and_connect_input_devices(self, interface, **kwargs):
        return self._build_and_connect_devices(interface, **kwargs)

    def _build_and_connect_output_devices(self, interface, **kwargs):
        interface["params"]["period"] = interface["params"].get("period", self.tvb_dt)
        return self._build_and_connect_devices(interface, **kwargs)

    def _default_receptor_type(self, source_node, target_node):
        return "exc"

    @property
    def _default_min_delay(self):
        return self.annarchy_min_delay


class ANNarchyInterfaceBuilder(ANNarchyProxyNodesBuilder, SpikeNetInterfaceBuilder):
    
    """ANNarchyInterfaceBuilder class"""

    _tvb_to_spikeNet_models = TVBtoANNarchyModels
    _spikeNet_to_tvb_models = ANNarchyToTVBModels

    _default_tvb_to_spikeNet_models = DefaultTVBtoANNarchyModels
    _default_spikeNet_to_tvb_models = DefaultANNarchytoTVBModels

    _input_proxy_models = ANNarchyInputProxyModels
    _output_proxy_models = ANNarchyOutputProxyModels

    _output_interface_type = ANNarchyOutputInterface
    _input_interface_type = ANNarchyInputInterface

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    logger = Attr(
        label="Logger",
        field_type=Logger,
        doc="""logging.Logger instance.""",
        required=True,
        default=initialize_logger(__name__, config=CONFIGURED)
    )


class ANNarchyRemoteInterfaceBuilder(ANNarchyInterfaceBuilder, SpikeNetRemoteInterfaceBuilder):

    """ANNarchyRemoteInterfaceBuilder class"""

    _output_interface_type = ANNarchySenderInterface
    _input_interface_type = ANNarchyReceiverInterface

    def configure(self):
        SpikeNetRemoteInterfaceBuilder.configure(self)


class ANNarchyTransformerInterfaceBuilder(ANNarchyInterfaceBuilder, SpikeNetTransformerInterfaceBuilder):

    """ANNarchyTransformerInterfaceBuilder class"""

    _output_interface_type = ANNarchyTransformerSenderInterface
    _input_interface_type = ANNarchyReceiverTransformerInterface

    def configure(self):
        SpikeNetTransformerInterfaceBuilder.configure(self)


class ANNarchyOutputTransformerInterfaceBuilder(ANNarchyInterfaceBuilder, SpikeNetOutputTransformerInterfaceBuilder):

    """ANNarchyOutputTransformerInterfaceBuilder class"""

    _input_proxy_models = ANNarchyInputProxyModels
    _output_proxy_models = ANNarchyOutputProxyModels

    _output_interface_type = ANNarchyTransformerSenderInterface
    _input_interface_type = ANNarchyReceiverInterface

    def configure(self):
        SpikeNetOutputTransformerInterfaceBuilder.configure(self)


class ANNarchyInputTransformerInterfaceBuilder(ANNarchyInterfaceBuilder, SpikeNetInputTransformerInterfaceBuilder):

    """ANNarchyInputTransformerInterfaceBuilder class"""

    _input_proxy_models = ANNarchyInputProxyModels
    _output_proxy_models = ANNarchyOutputProxyModels

    _output_interface_type = ANNarchySenderInterface
    _input_interface_type = ANNarchyReceiverTransformerInterface

    def configure(self):
        SpikeNetInputTransformerInterfaceBuilder.configure(self)


class TVBANNarchyInterfaceBuilder(ANNarchyProxyNodesBuilder, TVBSpikeNetInterfaceBuilder):

    """TVBANNarchyInterfaceBuilder class"""

    _tvb_to_spikeNet_models = TVBtoANNarchyModels
    _spikeNet_to_TVB_models = ANNarchyToTVBModels

    _default_ANNarchy_to_tvb_models = DefaultANNarchytoTVBModels
    _default_tvb_to_ANNarchy_models = DefaultTVBtoANNarchyModels

    _input_proxy_models = ANNarchyOutputProxyModels  # Input to SpikeNet is output of TVB
    _output_proxy_models = ANNarchyInputProxyModels  # Output of SpikeNet is input to TVB

    _output_interface_type = TVBtoANNarchyInterface
    _input_interface_type = ANNarchyToTVBInterface

    def configure(self):
        TVBSpikeNetInterfaceBuilder.configure(self)

    def _get_tvb_delays(self):
        return np.maximum(self.spiking_dt,
                          TVBSpikeNetInterfaceBuilder._get_tvb_delays(self) -
                          self.synchronization_time + self.tvb_dt)
