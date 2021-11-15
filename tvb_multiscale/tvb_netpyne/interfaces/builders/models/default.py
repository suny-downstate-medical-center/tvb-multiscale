from abc import ABCMeta, abstractmethod

import numpy as np

from tvb_multiscale.tvb_netpyne.interfaces.builders.base import TVBNetpyneInterfaceBuilder
from tvb_multiscale.tvb_netpyne.interfaces.base import TVBNetpyneInterface
from tvb_multiscale.tvb_netpyne.interfaces.models import RedWWexcIOinhI
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import random_normal_tvb_weight, \
    random_uniform_tvb_delay, receptor_by_source_region


class DefaultInterfaceBuilder(TVBNetpyneInterfaceBuilder):
    __metaclass__ = ABCMeta

    _tvb_netpyne_interface = RedWWexcIOinhI  # Set here the target interface model

    def __init__(self, tvb_simulator, netpyne_network, netpyne_nodes_ids, exclusive_nodes=False,
                 tvb_to_netpyne_interfaces=None, netpyne_to_tvb_interfaces=None, populations_sizes=[100, 100]):

        super(DefaultInterfaceBuilder, self).__init__(tvb_simulator, netpyne_network, netpyne_nodes_ids, exclusive_nodes,
                                                      tvb_to_netpyne_interfaces, netpyne_to_tvb_interfaces)
        self.N_E = populations_sizes[0]
        self.N_I = populations_sizes[1]

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        self.global_coupling_scaling = self.tvb_simulator.coupling.a[0].item()

    # By default we choose weights and delays with a random jitter around TVB ones!

    def tvb_weight_fun(self, source_node, target_node, scale=None, sigma=0.1):
        if scale is None:
            scale = self.global_coupling_scaling
        return random_normal_tvb_weight(source_node, target_node, self.tvb_weights, scale=scale, sigma=sigma)

    def tvb_delay_fun(self, source_node, target_node, low=None, high=None, sigma=0.1):
        if low is None:
            low = self.tvb_simulator.integrator.dt
        if high is None:
            high = 2*low
        return random_uniform_tvb_delay(source_node, target_node, self.tvb_delays, low, high, sigma)

    def receptor_fun(self, source_node, target_node, start=0):
        return 0

    # Spike rates are applied in parallelto neurons...

    def _build_default_rate_tvb_to_netpyne_interfaces(self, connections, **kwargs):
        # For spike transmission from TVB to NetPyNE devices as TVB proxy nodes with TVB delays:
        # Options:
        # "model": "poisson_generator", "params": {"allow_offgrid_times": False}
        # For spike trains with correlation probability p_copy set:
        # "model": "mip_generator", "params": {"p_copy": 0.5, "mother_seed": 0}
        # TODO: above - proper model names (e.g. proper for spike_generator_placeholder instead of inhomogeneous_poisson_generator)

        interface = \
            {"model": "spike_generator_placeholder",
             "params": {"allow_offgrid_times": False},
        # -------Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)---------
              "interface_weights": 1.0*self.N_E,
        # Applied outside NetPyNE for each interface device
        #                                  Function of TVB connectivity weight:
              "weights": self.tvb_weight_fun,
        #                                  Function of TVB connectivity delay:
              "delays": self.tvb_delay_fun,
              "receptor_type": self.receptor_fun,
              "neurons_inds": lambda tvb_id, netpyne_id, neurons_inds:
                                tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
        # --------------------------------------------------------------------------------------------------------------
        #                           TVB sv or param -> NetPyNE population
              "connections": connections,
              "source_nodes": None, "target_nodes": None}  # None means all here
        interface.update(kwargs)
        self.tvb_to_spikeNet_interfaces.append(interface)

    # ...unlike currents that have to be distributed to neurons (e.g., N_E / N_E = 1.0)

    def _build_default_current_tvb_to_netpyne_interfaces(self, connections, **kwargs):
        # For injecting current to NetPyNE neurons via dc generators acting as TVB proxy nodes with TVB delays:
        interface = \
            {"model": "dc_generator", "params": {},
             # -------Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)---------
             # Applied outside NetPyNE for each interface device:
             "interface_weights": 1.0,
             #                                 Function of TVB connectivity weight:
             "weights": self.tvb_weight_fun,
             #                                 Function of TVB connectivity delay:
             "delays": self.tvb_delay_fun,
             "neurons_inds": lambda tvb_id, netpyne_id, neurons_inds:
                                tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
             # --------------------------------------------------------------------------------------------------------------
             #                                                 TVB sv -> NetPyNE population
             "connections": connections,
             "source_nodes": None, "target_nodes": None}  # None means all here
        interface.update(kwargs)
        self.tvb_to_spikeNet_interfaces.append(interface)

    def _build_default_param_tvb_to_netpyne_interfaces(self, connections, **kwargs):
        # For directly setting an external current parameter in NetPyNE neurons instantaneously:
        interface = \
            {"model": "current", "parameter": "I_e",
            # ---------Properties potentially set as function handles with args (netpyne_node_id=None)-------------------------
            "interface_weights": 1.0,
            "neurons_inds": lambda node_id, neurons_inds:
                                tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
            # --------------------------------------------------------------------------------------------------------------
            #                                               TVB sv -> NetPyNE population
            "connections": connections,
            "nodes": None}  # None means all here
        interface.update(kwargs)
        self.tvb_to_spikeNet_interfaces.append(interface)

    def _build_default_netpyne_to_tvb_interfaces(self, connections, **kwargs):
        # NetPyNE -> TVB:
        interface = \
            {"model": "spike_recorder", "params": {},
             # ------------------Properties potentially set as function handles with args (netpyne_node_id=None)----------------
             "interface_weights": 1.0, "delays": 0.0,
             "neurons_inds": lambda node_id, neurons_inds:
                                 tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
             # --------------------------------------------------------------------------------------------------------------
             "connections": connections, "nodes": None}  # None means all here
        interface.update(kwargs)
        self.spikeNet_to_tvb_interfaces.append(interface)

    @abstractmethod
    def build_default_rate_tvb_to_netpyne_interfaces(self):
        raise NotImplementedError

    @abstractmethod
    def build_default_current_tvb_to_netpyne_interfaces(self):
        raise NotImplementedError

    @abstractmethod
    def build_default_param_tvb_to_netpyne_interfaces(self):
        raise NotImplementedError

    @abstractmethod
    def build_default_netpyne_to_tvb_interfaces(self):
        raise NotImplementedError

    def default_build(self, tvb_to_netpyne_mode="rate", netpyne_to_tvb=True):
        if tvb_to_netpyne_mode and \
                (self.tvb_to_spikeNet_interfaces is None or len(self.tvb_to_spikeNet_interfaces) == 0):
            self.tvb_to_spikeNet_interfaces = []
            if tvb_to_netpyne_mode.lower() == "rate":
                # For spike transmission from TVB to NetPyNE devices as TVB proxy nodes with TVB delays:
                self.build_default_rate_tvb_to_netpyne_interfaces()
            elif tvb_to_netpyne_mode.lower() == "current":
                # For injecting current to NetPyNE neurons via dc generators acting as TVB proxy nodes with TVB delays:
                self.build_default_current_tvb_to_netpyne_interfaces()
            elif tvb_to_netpyne_mode.lower() == "param":
                # For directly setting an external current parameter in NetPyNE neurons instantaneously:
                self.build_default_param_tvb_to_netpyne_interfaces()

        # The NetPyNE nodes the activity of which is transformed to TVB state variables or parameters
        if netpyne_to_tvb and \
                (self.spikeNet_to_tvb_interfaces is None or len(self.spikeNet_to_tvb_interfaces) == 0):
            self.spikeNet_to_tvb_interfaces = []
            self.build_default_netpyne_to_tvb_interfaces()

    def build_interface(self, tvb_netpyne_interface=None, tvb_to_netpyne_mode="rate", netpyne_to_tvb=True):
        self.default_build(tvb_to_netpyne_mode, netpyne_to_tvb)
        if not isinstance(tvb_netpyne_interface, TVBNetpyneInterface):
            tvb_netpyne_interface = self._tvb_netpyne_interface()
        return super(DefaultInterfaceBuilder, self).build_interface(tvb_netpyne_interface)


class DefaultMultiSynapseInterfaceBuilder(DefaultInterfaceBuilder):
    __metaclass__ = ABCMeta

    def receptor_fun(self, source_node, target_node, start=3):
        return receptor_by_source_region(source_node, target_node, start)

    @abstractmethod
    def build_default_rate_tvb_to_netpyne_interfaces(self):
        raise NotImplementedError

    @abstractmethod
    def build_default_current_tvb_to_netpyne_interfaces(self):
        raise NotImplementedError

    @abstractmethod
    def build_default_param_tvb_to_netpyne_interfaces(self):
        raise NotImplementedError

    @abstractmethod
    def build_default_netpyne_to_tvb_interfaces(self):
        raise NotImplementedError