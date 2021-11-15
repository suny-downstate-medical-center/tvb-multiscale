import numpy as np

from tvb_multiscale.core.spiking_models.builders.base import SpikingModelBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices

from tvb_multiscale.tvb_netpyne.netpyne_models.network import NetpyneNetwork
from tvb_multiscale.tvb_netpyne.netpyne_models.population import NetpynePopulation
from tvb_multiscale.tvb_netpyne.ext.instance import NetpyneInstance
from tvb_multiscale.tvb_netpyne.config import CONFIGURED, initialize_logger

from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import create_device, connect_device
from tvb_multiscale.tvb_netpyne.netpyne_models.region_node import NetpyneRegionNode
from tvb_multiscale.tvb_netpyne.netpyne_models.brain import NetpyneBrain

LOG = initialize_logger(__name__)

class NetpyneModelBuilder(SpikingModelBuilder):

    config = CONFIGURED
    netpyne_instance = None
    modules_to_install = []
    _spiking_brain = NetpyneBrain()

    def __init__(self, tvb_simulator, spiking_nodes_ids, netpyne_instance=None, config=CONFIGURED, logger=LOG):
        super(NetpyneModelBuilder, self).__init__(tvb_simulator, spiking_nodes_ids, config, logger)
        self.netpyne_instance = netpyne_instance
        self._spiking_brain =  NetpyneBrain()

    def configure(self):
        self.netpyne_instance = NetpyneInstance()
        super(NetpyneModelBuilder, self).configure()
        # self.compile_install_nest_modules(self.modules_to_install)
        # self.confirm_compile_install_nest_models(self._models)

    def set_synapse(self, syn_model, weight, delay, receptor_type, params={}):
        """Method to set the synaptic model, the weight, the delay,
           the synaptic receptor type, and other possible synapse parameters
           to a synapse_params dictionary.
           Arguments:
            - syn_model: the name (string) of the synapse model
            - weight: the weight of the synapse
            - delay: the delay of the connection,
            - receptor_type: the receptor type
            - params: a dict of possible synapse parameters
           Returns:
            a dictionary of the whole synapse configuration
        """
        syn_spec = {'synapse_model': syn_model, 'weight': weight, 'delay': delay, 'receptor_type': receptor_type}
        syn_spec.update(params)
        return syn_spec

    node_creation_iteration = 0
    def build_spiking_population(self, pop_label, model, size, params):
        """This methods builds a NetpynePopulation instance,
           which represents a population of spiking neurons of the same neural model,
           and residing at a particular brain region node.
           Arguments:
            label: name (string) of the population
            model: name (string) of the neural model
            size: number (integer) of the neurons of this population
            params: dictionary of parameters of the neural model to be set upon creation
           Returns:
            a NetpynePopulation class instance
        """
        # TODO: very hacky way..
        # TODO.TVB: direct way to get node id??
        if self.node_creation_iteration >= 2:
            index = 1
        else:
            index = 0
        node_label = self.spiking_nodes_labels[index]
        self.node_creation_iteration += 1

        size = int(np.round(size))

        # node collection for current spiking node
        node_collection = self.netpyne_instance.createNodeCollection(node_label, pop_label, model, size, params=params)
        # populations for artifitial cells representing connections from external nodes
        self._create_artificial_cells(size, project_to_node=node_label, project_to_pop=pop_label)

        label = node_collection.label
        return NetpynePopulation(node_collection, label, model, self.netpyne_instance)

    def _create_artificial_cells(self, size, project_to_node, project_to_pop):
        for node_id, node_label in enumerate(self.tvb_connectivity.ordered_labels):
            if node_id not in self.spiking_nodes_ids:
                self.netpyne_instance.createArtificialCells(self.state_variable, node_label, project_to_node, project_to_pop, size, params=None)

    def connect_two_populations(self, pop_src, src_inds_fun, pop_trg, trg_inds_fun, conn_spec, syn_spec):
        """Method to connect two NetpynePopulation instances in the SpikingNetwork.
           Arguments:
            source: the source NetpynePopulation of the connection
            src_inds_fun: a function that selects a subset of the souce population neurons
            target: the target NetpynePopulation of the connection
            trg_inds_fun: a function that selects a subset of the target population neurons
            conn_params: a dict of parameters of the connectivity pattern among the neurons of the two populations,
                         excluding weight and delay ones
            synapse_params: a dict of parameters of the synapses among the neurons of the two populations,
                            including weight, delay and synaptic receptor type ones
        """
        # TODO: parse also conn_spec. And should we also use syn_spec["receptor_type"], src_inds_fun, trg_inds_fun? (see NEST for reference)
        self.netpyne_instance.createInternalConnection(pop_src.label, pop_trg.label, syn_spec["synapse_model"], syn_spec["weight"], syn_spec["delay"])

    def build_spiking_region_node(self, label="", input_node=None, *args, **kwargs):
        """This methods builds a NetpyneRegionNode instance,
           which consists of a pandas.Series of all SpikingPopulation instances,
           residing at a particular brain region node.
           Arguments:
            label: name (string) of the region node. Default = ""
            input_node: an already created SpikingRegionNode() class. Default = None.
            *args, **kwargs: other optional positional or keyword arguments
           Returns:
            a SpikingRegionNode class instance
        """
        return NetpyneRegionNode(label, input_node, self.netpyne_instance)

    def build_and_connect_devices(self, devices):
        """Method to build and connect input or output devices, organized by
           - the variable they measure or stimulate (pandas.Series), and the
           - population(s) (pandas.Series), and
           - brain region nodes (pandas.Series) they target.
           See tvb_multiscale.core.spiking_models.builders.factory
           and tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory"""
        # TODO: check the following assumption: this part is used only to record spiking activity for observation, not for updating TVB state
        return build_and_connect_devices(devices, create_device, connect_device,
                                         self._spiking_brain, self.config, netpyne_instance=self.netpyne_instance)

    def build_spiking_region_nodes(self, *args, **kwargs):
        super(NetpyneModelBuilder, self).build_spiking_region_nodes(*args, **kwargs)
        self.netpyne_instance.createCells()

    def build(self):
        """A method to build the final NetpyneNetwork class based on the already created constituents."""
        return NetpyneNetwork(self.netpyne_instance, self._spiking_brain,
                              self._output_devices, self._input_devices, config=self.config)