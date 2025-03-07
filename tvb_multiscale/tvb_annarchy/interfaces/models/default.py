# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.default import DefaultTVBSpikeNetInterfaceBuilder, \
    DefaultSpikeNetRemoteInterfaceBuilder, DefaultSpikeNetTransformerInterfaceBuilder, \
    DefaultSpikeNetOutputTransformerInterfaceBuilder, DefaultSpikeNetInputTransformerInterfaceBuilder, \
    DefaultSpikeNetInterfaceBuilder, DefaultSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_annarchy.interfaces.builders import ANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder, \
    ANNarchyRemoteInterfaceBuilder, TVBANNarchyInterfaceBuilder, \
    ANNarchyTransformerInterfaceBuilder, ANNarchyOutputTransformerInterfaceBuilder, \
    ANNarchyInputTransformerInterfaceBuilder


class DefaultANNarchyProxyNodesBuilder(ANNarchyProxyNodesBuilder, DefaultSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class DefaultANNarchyInterfaceBuilder(DefaultANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder,
                                      DefaultSpikeNetInterfaceBuilder):
    pass


class DefaultANNarchyRemoteInterfaceBuilder(DefaultANNarchyInterfaceBuilder, ANNarchyRemoteInterfaceBuilder,
                                        DefaultSpikeNetRemoteInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class DefaultNESNetOutputTransformerInterfaceBuilder(DefaultANNarchyInterfaceBuilder,
                                                     ANNarchyOutputTransformerInterfaceBuilder,
                                                     DefaultSpikeNetOutputTransformerInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetOutputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetOutputTransformerInterfaceBuilder.default_input_config(self)


class DefaultANNarchyInputTransformerInterfaceBuilder(DefaultANNarchyInterfaceBuilder,
                                                      ANNarchyInputTransformerInterfaceBuilder,
                                                      DefaultSpikeNetInputTransformerInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetInputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetInputTransformerInterfaceBuilder.default_input_config(self)


class DefaultANNarchyTransformerInterfaceBuilder(DefaultANNarchyInterfaceBuilder, ANNarchyTransformerInterfaceBuilder,
                                             DefaultSpikeNetTransformerInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class DefaultTVBANNarchyInterfaceBuilder(DefaultANNarchyProxyNodesBuilder, TVBANNarchyInterfaceBuilder,
                                     DefaultTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_input_config(self)
