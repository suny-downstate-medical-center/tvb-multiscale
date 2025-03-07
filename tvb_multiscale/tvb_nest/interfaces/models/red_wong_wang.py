# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.red_wong_wang import \
    RedWongWangExcIOTVBSpikeNetInterfaceBuilder, \
    RedWongWangExcIOSpikeNetRemoteInterfaceBuilder, RedWongWangExcIOSpikeNetTransformerInterfaceBuilder, \
    RedWongWangExcIOSpikeNetOutputTransformerInterfaceBuilder, \
    RedWongWangExcIOSpikeNetInputTransformerInterfaceBuilder, \
    RedWongWangExcIOSpikeNetInterfaceBuilder, RedWongWangExcIOSpikeNetProxyNodesBuilder, \
    RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder, \
    RedWongWangExcIOInhISpikeNetRemoteInterfaceBuilder, RedWongWangExcIOInhISpikeNetTransformerInterfaceBuilder, \
    RedWongWangExcIOInhISpikeNetOutputTransformerInterfaceBuilder, \
    RedWongWangExcIOInhISpikeNetInputTransformerInterfaceBuilder, \
    RedWongWangExcIOInhISpikeNetInterfaceBuilder, RedWongWangExcIOInhISpikeNetProxyNodesBuilder


from tvb_multiscale.tvb_nest.interfaces.builders import NESTProxyNodesBuilder, NESTInterfaceBuilder, \
    NESTRemoteInterfaceBuilder, TVBNESTInterfaceBuilder, \
    NESTTransformerInterfaceBuilder, NESTOutputTransformerInterfaceBuilder, NESTInputTransformerInterfaceBuilder

from tvb_multiscale.tvb_nest.nest_models.builders.nest_templates import receptor_by_source_region


class RedWongWangExcIONESTProxyNodesBuilder(NESTProxyNodesBuilder, RedWongWangExcIOSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class RedWongWangExcIONESTInterfaceBuilder(RedWongWangExcIONESTProxyNodesBuilder, NESTInterfaceBuilder,
                                           RedWongWangExcIOSpikeNetInterfaceBuilder):
    pass


class RedWongWangExcIONESTRemoteInterfaceBuilder(RedWongWangExcIONESTInterfaceBuilder, NESTRemoteInterfaceBuilder,
                                                 RedWongWangExcIOSpikeNetRemoteInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class RedWongWangExcIONESTOutputTransformerInterfaceBuilder(
    RedWongWangExcIONESTInterfaceBuilder, NESTOutputTransformerInterfaceBuilder,
    RedWongWangExcIOSpikeNetOutputTransformerInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOSpikeNetOutputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOSpikeNetOutputTransformerInterfaceBuilder.default_input_config(self)


class RedWongWangExcIONESTInputTransformerInterfaceBuilder(RedWongWangExcIONESTInterfaceBuilder,
                                                           NESTInputTransformerInterfaceBuilder,
                                                           RedWongWangExcIOSpikeNetInputTransformerInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOSpikeNetInputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOSpikeNetInputTransformerInterfaceBuilder.default_input_config(self)


class RedWongWangExcIONESTTransformerInterfaceBuilder(RedWongWangExcIONESTInterfaceBuilder,
                                                      NESTTransformerInterfaceBuilder,
                                                      RedWongWangExcIOSpikeNetTransformerInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOTVBNESTInterfaceBuilder(RedWongWangExcIONESTProxyNodesBuilder, TVBNESTInterfaceBuilder,
                                              RedWongWangExcIOTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOTVBSpikeNetInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOInhINESTProxyNodesBuilder(NESTProxyNodesBuilder,
                                                RedWongWangExcIOInhISpikeNetProxyNodesBuilder):

    def _default_receptor_type(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=1)


class RedWongWangExcIOInhINESTInterfaceBuilder(RedWongWangExcIOInhINESTProxyNodesBuilder, NESTInterfaceBuilder,
                                               RedWongWangExcIOInhISpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class RedWongWangExcIOInhINESTRemoteInterfaceBuilder(RedWongWangExcIOInhINESTInterfaceBuilder,
                                                     NESTRemoteInterfaceBuilder,
                                                     RedWongWangExcIOInhISpikeNetRemoteInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOInhISpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOInhISpikeNetRemoteInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOInhINESTOutputTransformerInterfaceBuilder(
    RedWongWangExcIOInhINESTInterfaceBuilder, NESTOutputTransformerInterfaceBuilder,
    RedWongWangExcIOInhISpikeNetOutputTransformerInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOInhISpikeNetOutputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOInhISpikeNetOutputTransformerInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOInhINESTInputTransformerInterfaceBuilder(
    RedWongWangExcIOInhINESTInterfaceBuilder, NESTInputTransformerInterfaceBuilder,
    RedWongWangExcIOInhISpikeNetInputTransformerInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOInhISpikeNetInputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOInhISpikeNetInputTransformerInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOInhINESTTransformerInterfaceBuilder(RedWongWangExcIOInhINESTInterfaceBuilder,
                                                          NESTTransformerInterfaceBuilder,
                                                          RedWongWangExcIOInhISpikeNetTransformerInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOInhISpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOInhISpikeNetTransformerInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOInhITVBNESTInterfaceBuilder(RedWongWangExcIOInhINESTProxyNodesBuilder, TVBNESTInterfaceBuilder,
                                                  RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder.default_input_config(self)
