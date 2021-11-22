import numpy as np

from tvb_multiscale.core.spiking_models.builders.base import SpikingNetworkBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices

from tvb_multiscale.tvb_netpyne.netpyne_models.network import NetpyneNetwork
from tvb_multiscale.tvb_netpyne.netpyne_models.population import NetpynePopulation
from tvb_multiscale.tvb_netpyne.ext.instance import NetpyneInstance
from tvb_multiscale.tvb_netpyne.config import CONFIGURED, initialize_logger

from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import create_device, connect_device
from tvb_multiscale.tvb_netpyne.netpyne_models.region_node import NetpyneRegionNode
from tvb_multiscale.tvb_netpyne.netpyne_models.brain import NetpyneBrain

LOG = initialize_logger(__name__)

class NetpyneNetworkBuilder(SpikingNetworkBuilder):

    config = CONFIGURED
    netpyne_instance = None
    modules_to_install = []
    _spiking_brain = NetpyneBrain()

    def __init__(self, tvb_simulator, spiking_nodes_ids, netpyne_instance=None, config=CONFIGURED, logger=LOG):
        super(NetpyneNetworkBuilder, self).__init__(tvb_simulator, spiking_nodes_ids, config, logger)
        self.netpyne_instance = netpyne_instance
        self._spiking_brain =  NetpyneBrain()

    def configure(self):
        self.netpyne_instance = NetpyneInstance()
        super(NetpyneNetworkBuilder, self).configure()
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

    def build_spiking_population(self, label, model, brain_region, size, params):
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
        # TODO: parse low-level values from `params`
        pop_label = label

        size = int(np.round(size))

        # node collection for current spiking node
        node_collection = self.netpyne_instance.createNodeCollection(brain_region, pop_label, model, size, params=params)

        # TODO: need to do this in way described in build_and_connect_devices.. For now:
        if pop_label == 'E': # this test is to avoid creating them twice
            # populations for artifitial cells representing connections from external nodes
            self._create_artificial_cells(size, project_to_node=brain_region, project_to_pop=pop_label)
        return NetpynePopulation(node_collection, self.netpyne_instance, pop_label, model, brain_region)
        # TODO: needed?
        return NESTPopulation(self.nest_instance.Create(model, n_neurons, params=params),
                              nest_instance=self.nest_instance, label=label, model=model, brain_region=brain_region)

    def _create_artificial_cells(self, size, project_to_node, project_to_pop): 
        for region in self.region_labels:
            if region not in self.spiking_nodes_labels:
                label = self.state_variable + " - " + region # TODO: de-hardcode/workaround this composition
                self.netpyne_instance.createArtificialCells(label, size)

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
        # TODO: Parse also conn_spec. And should we also use syn_spec["receptor_type"], src_inds_fun, trg_inds_fun? (see NEST for reference)
        src = pop_src.brain_region + '.' + pop_src.label
        trg = pop_trg.brain_region + '.' + pop_trg.label
        self.netpyne_instance.createInternalConnection(src, trg, syn_spec["synapse_model"], syn_spec["weight"], syn_spec["delay"])

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
        return NetpyneRegionNode(input_node, label=label)

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
        super(NetpyneNetworkBuilder, self).build_spiking_region_nodes(*args, **kwargs)
        self.netpyne_instance.createCells()

    def build_spiking_network(self):
        """A method to build the final NetpyneNetwork class based on the already created constituents."""
        return NetpyneNetwork(self.netpyne_instance, self._spiking_brain,
                           self._output_devices, self._input_devices, config=self.config)