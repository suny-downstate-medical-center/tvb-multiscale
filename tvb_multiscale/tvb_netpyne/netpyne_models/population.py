from tvb_multiscale.core.spiking_models.population import SpikingPopulation

class NetpynePopulation(SpikingPopulation):

    netpyne_instance = None

    def __init__(self, node_collection, label, model, netpyne_instance = None):
        self.netpyne_instance = netpyne_instance
        super(NetpynePopulation, self).__init__(node_collection, label, model)

    def _print_neurons(self):
        pass

    @property
    def neurons(self):
        """Method to get a sequence (list, tuple, array) of the individual gids of populations' neurons"""
        gids = self.netpyne_instance.cellGids(self._population.label)
        return gids

    def _Set(self, values_dict, neurons=None):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
        """
        pass

    def _Get(self, attr=None, neurons=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of neurons' attributes.
        """
        pass

    def _GetConnections(self, neurons=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingPopulation neuron.
           Arguments:
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            connections' objects.
        """
        pass

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: connections' objects.
                          Default = None, corresponding to all connections to/from the present population.
        """
        pass

    def _GetFromConnections(self, attr=None, connections=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                    Default = None, corresponding to all attributes
             connections: connections' objects.
                          Default = None, corresponding to all connections to/from the present population.
            Returns:
             Dictionary of sequences (lists, tuples, or arrays) of connections' attributes.

        """
        pass