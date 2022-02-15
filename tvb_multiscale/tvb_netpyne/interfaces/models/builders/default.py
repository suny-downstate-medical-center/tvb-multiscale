from abc import ABCMeta, abstractmethod

import numpy as np
from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear

from tvb_multiscale.tvb_netpyne.interfaces.builders.base import TVBNetpyneInterfaceBuilder
from tvb_multiscale.tvb_netpyne.interfaces.base import TVBNetpyneInterface
from tvb_multiscale.tvb_netpyne.interfaces.models.models import RedWWexcIOinhI
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import random_normal_tvb_weight, \
    random_uniform_tvb_delay, receptor_by_source_region # TODO: is receptor_by_source_region needed?


class DefaultInterfaceBuilder(TVBNetpyneInterfaceBuilder):
    __metaclass__ = ABCMeta

    _tvb_netpyne_interface = RedWWexcIOinhI # TODO: why this one?

    def __init__(self, tvb_simulator, netpyne_network, netpyne_nodes_ids, exclusive_nodes=False,
                 tvb_to_netpyne_interfaces=None, netpyne_to_tvb_interfaces=None, populations_sizes=[100, 100]):

        super(DefaultInterfaceBuilder, self).__init__(tvb_simulator, netpyne_network, netpyne_nodes_ids, exclusive_nodes,
                                                      tvb_to_netpyne_interfaces, netpyne_to_tvb_interfaces)
        
        # TODO: not sure it has to be hardcoded this way. At least, default value [100, 100] may be delusive
        self.N_E = populations_sizes[0]
        self.N_I = populations_sizes[1]

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        self.global_coupling_scaling = self.tvb_simulator.coupling.a[0].item()

        self.lamda = 0.0
        # self.lamda = self.tvb_simulator.lamda[0].item() # TODO: find out how we can have this lamda in tvb_model

        # WongWang model parameter r is in Hz, just like poisson_generator assumes in NetPyNE:
        self.w_tvb_to_spike_rate = 1.0

        # given WongWang model parameter r is in Hz but tvb dt is in ms:
        self.w_spikes_to_tvb = 1000.0 / self.tvb_dt

    @abstractmethod
    def build_default_rate_tvb_to_netpyne_interfaces(self):
        state_variable = self.spiking_network.state_variable
        self._build_default_rate_tvb_to_netpyne_interfaces({state_variable: ["E"]})
        if self.lamda > 0:
             self._build_default_rate_tvb_to_netpyne_interfaces({state_variable: ["I"]}, scale=self.lamda)

    @abstractmethod
    def build_default_netpyne_to_tvb_interfaces(self):
        from collections import OrderedDict
        connections = OrderedDict()
        connections["Rin_e"] = ["E"]
        connections["Rin_i"] = ["I"]
        self._build_default_netpyne_to_tvb_interfaces(connections)

    # By default we choose weights and delays with a random jitter around TVB ones!

    def tvb_weight_fun(self, source_node, target_node, scale=None, sigma=0.1):
        if scale is None:
            scale = 10 * 2 * self.global_coupling_scaling #TODO: this is not proper scaling. take from DefaultExcIOInhIBuilder (or see #NOTE!!! above). and dehardcode this 10 (see tvb_to_netpyne_weight_scaling)
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

    def _build_default_rate_tvb_to_netpyne_interfaces(self, connections, scale=1.0, **kwargs):
        # For spike transmission from TVB to NetPyNE devices as TVB proxy nodes with TVB delays:
        # Options:
        # "model": "poisson_generator", "params": {"allow_offgrid_times": False}

        interface = \
            {"model": "poisson_generator",
             "params": {"connectivity_scale": scale},
        # -------Properties potentially set as function handles with args (tvb_node_id=None, netpyne_node_id=None)---------
              "interface_weights": 1.0 * self.N_E,
        # Applied outside NetPyNE for each interface device
        #                                  Function of TVB connectivity weight:
              "weights": self.tvb_weight_fun,
        #                                  Function of TVB connectivity delay:
              "delays": self.tvb_delay_fun,
              "receptor_type": self.receptor_fun,
            #   "neurons_inds": lambda tvb_id, netpyne_id, neurons_inds:
            #                     tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
        # --------------------------------------------------------------------------------------------------------------
        #                           TVB sv or param -> NetPyNE population
              "connections": connections,
              "source_nodes": None, "target_nodes": None}  # None means all here
        interface.update(kwargs)
        self.tvb_to_spikeNet_interfaces.append(interface)

    def _build_default_netpyne_to_tvb_interfaces(self, connections, **kwargs):
        # NetPyNE -> TVB:
        interface = \
            {"model": "spike_recorder", "params": {},
             # ------------------Properties potentially set as function handles with args (netpyne_node_id=None)----------------
             "interface_weights": 1.0, "delays": 0.0,
            #  "neurons_inds": lambda node_id, neurons_inds:
            #                      tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
             # --------------------------------------------------------------------------------------------------------------
             "connections": connections, "nodes": None}  # None means all here
        interface.update(kwargs)
        self.spikeNet_to_tvb_interfaces.append(interface)

    def default_build(self, tvb_to_netpyne_mode="rate", netpyne_to_tvb=True):
        if tvb_to_netpyne_mode and \
                (self.tvb_to_spikeNet_interfaces is None or len(self.tvb_to_spikeNet_interfaces) == 0):
            # TODO: condition never met. Should it? Also, revise most of methods in this class, are they needed?
            self.tvb_to_spikeNet_interfaces = []
            if tvb_to_netpyne_mode.lower() == "rate":
                # For spike transmission from TVB to NetPyNE devices as TVB proxy nodes with TVB delays:
                self.build_default_rate_tvb_to_netpyne_interfaces()
            else:
                raise NotImplementedError("Only 'rate' interface is available")

        # The NetPyNE nodes the activity of which is transformed to TVB state variables or parameters
        if netpyne_to_tvb and \
                (self.spikeNet_to_tvb_interfaces is None or len(self.spikeNet_to_tvb_interfaces) == 0):
            self.spikeNet_to_tvb_interfaces = []
            self.build_default_netpyne_to_tvb_interfaces()

    def build_interface(self, tvb_spikeNet_interface=None, tvb_to_spikeNet_mode="rate", spikeNet_to_tvb=True):
        self.default_build(tvb_to_spikeNet_mode, spikeNet_to_tvb)
        if not isinstance(tvb_spikeNet_interface, TVBNetpyneInterface):
            tvb_spikeNet_interface = self._tvb_netpyne_interface()
        super(DefaultInterfaceBuilder, self).build_interface(tvb_spikeNet_interface)

        total_scale = None
        for tvb_to_spike in tvb_spikeNet_interface.tvb_to_spikeNet_interfaces:
            if total_scale is None:
                total_scale = tvb_to_spike.scale
            else:
                total_scale += tvb_to_spike.scale

        for spike_to_tvb in tvb_spikeNet_interface.spikeNet_to_tvb_interfaces:
            spike_to_tvb.tvb_dt = tvb_spikeNet_interface.dt # TODO: might not be needed, if we pass dt to NetpyneInstance
        for tvb_to_spike in tvb_spikeNet_interface.tvb_to_spikeNet_interfaces:
            tvb_to_spike.total_scale = total_scale

            # TODO: investigate the exception when trying to print weights here
            # print(tvb_to_spike.weights)

        return tvb_spikeNet_interface

class DefaultMultiSynapseInterfaceBuilder(DefaultInterfaceBuilder):
    __metaclass__ = ABCMeta

    def receptor_fun(self, source_node, target_node, start=3):
        return receptor_by_source_region(source_node, target_node, start)

    @abstractmethod
    def build_default_rate_tvb_to_netpyne_interfaces(self):
        raise NotImplementedError

    @abstractmethod
    def build_default_netpyne_to_tvb_interfaces(self):
        raise NotImplementedError