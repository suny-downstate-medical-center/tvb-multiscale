from tvb_multiscale.core.config import CONFIGURED
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneModelBuilder

class DefaultExcIOInhIBuilder(NetpyneModelBuilder):

    def __init__(self, tvb_simulator, spiking_nodes_ids, netpyne_instance=None, config=CONFIGURED):
        super(DefaultExcIOInhIBuilder, self).__init__(tvb_simulator, spiking_nodes_ids, netpyne_instance=netpyne_instance, config=config)
        self.scale_e = 1
        self.scale_i = 1
    
    def set_defaults(self):
        # TODO:
        # self.set_populations()
        # self.set_populations_connections()
        # self.set_nodes_connections()
        # self.set_output_devices()
        # self.set_input_devices()
        pass

    # def receptor_E_fun(self):
    #     return 0

    # def receptor_I_fun(self):
    #     return 0

    # def set_EE_populations_connections(self):
    #     connections = \
    #         {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
    #          "synapse_model": self.default_populations_connection["synapse_model"],
    #          "conn_spec": self.default_populations_connection["conn_spec"],
    #          "weight": self.w_ee,
    #          "delay": self.d_ee,
    #          "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
    #     connections.update(self.pop_conns_EE)
    #     return connections

    # def set_EI_populations_connections(self):
    #     connections = \
    #         {"source": "E", "target": "I",  # E -> I
    #          "synapse_model": self.default_populations_connection["synapse_model"],
    #          "conn_spec": self.default_populations_connection["conn_spec"],
    #          "weight": self.w_ei,
    #          "delay": self.d_ei,
    #          "receptor_type": self.receptor_E_fun(), "nodes": None}  # None means "all"
    #     connections.update(self.pop_conns_EI)
    #     return connections

    # def set_IE_populations_connections(self):
    #     connections = \
    #         {"source": "I", "target": "E",  # I -> E
    #          "synapse_model": self.default_populations_connection["synapse_model"],
    #          "conn_spec": self.default_populations_connection["conn_spec"],
    #          "weight": self.w_ie,
    #          "delay": self.d_ie,
    #          "receptor_type": self.receptor_I_fun(), "nodes": None}  # None means "all"
    #     connections.update(self.pop_conns_IE)
    #     return connections

    # def set_II_populations_connections(self):
    #     connections = \
    #         {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
    #          "synapse_model": self.default_populations_connection["synapse_model"],
    #          "conn_spec": self.default_populations_connection["conn_spec"],
    #          "weight": self.w_ii,
    #          "delay": self.d_ii,
    #          "receptor_type": self.receptor_I_fun(), "nodes": None}  # None means "all"
    #     connections.update(self.pop_conns_II)
    #     return connections

    def set_populations_connections(self):
        pass
    #     self.populations_connections = [
    #        self.set_EE_populations_connections(), self.set_EI_populations_connections(),
    #        self.set_IE_populations_connections(), self.set_II_populations_connections()
    #     ]