import neuron
import numpy
from tvb_multiscale.core.spiking_models.devices import Device, InputDevice, OutputDevice, SpikeRecorder
from tvb.basic.neotraits.api import HasTraits, Attr, Int, List

class NetpyneDevice(HasTraits):

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        self.netpyne_instance = netpyne_instance
        self.device__ = device # TODO: find out why inherited `device` property didn't work out here
        HasTraits.__init__(self)
    
    def _print_neurons(self):
        pass

    def _assert_device(self):
        """Method to assert that the node of the network is a device"""
        pass

    def _assert_spiking_simulator(self):
        pass

    @property
    def gids(self):
        """Method to get a sequence (list, tuple, array) of the individual gids of nodes's elements"""
        pass

    def _assert_nodes(self, nodes=None):
        """Method to assert that the node of the network is valid"""
        pass

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the device
            Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: connections' objects. Default = None, corresponding to all device's connections
        """
        pass

    def _GetFromConnections(self, attrs=None, connections=None):
        """Method to get attributes of the connections from/to the device
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all device's attributes
            connections: connections' objects. Default = None, corresponding to all device's connections
           Returns:
            Dictionary of sequences (tuples, lists, arrays) of connections' attributes.
        """
        pass

    # @abstractmethod
    def _Set(self, values_dict, nodes=None):
        """Method to set attributes of the SpikingNodeCollection's nodes.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            nodes: instance of a nodes class,
                   or sequence (list, tuple, array) of nodes the attributes of which should be set.
                   Default = None, corresponds to all nodes.
        """
        if self.model == "poisson_generator":
            raise NotImplementedError
        elif self.model == "spike_generator_placeholder":
            self.device__.applyFiringRate(values_dict, self.label)

    def _Get(self, attr=None, nodes=None):
        """Method to get attributes of the SpikingNodeCollection's nodes.
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of nodes' attributes.
        """
        pass

    # @abstractmethod
    def _GetConnections(self, nodes=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingNodeCollection node.
           Arguments:
            nodes: instance of a nodes class,
                   or sequence (list, tuple, array) of nodes the attributes of which should be set.
                   Default = None, corresponds to all nodes.
            source_or_target: Direction of connections relative to nodes
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            connections' objects.
        """
        # So far, tvb implementation is interested only in number of connections.
        # Thus it's equivalent to number of neurons this device is connected to/from.
        return self.neurons

    # @property
    # @abstractmethod
    # def connections(self):
    #     """Method to get all connections of the device to/from the device.
    #        Returns:
    #         connections' objects.
    #     """
    #     pass

    # @property
    # def neurons(self):
    #     """Method to get the indices of all the neurons the device is connected to/from."""
    #     pass # overriden in subclasses

    def get_neurons(self):
        """Method to get the indices of all the neurons the device is connected to/from."""
        return self.neurons

class NetpyneInputDevice(NetpyneDevice, InputDevice):

    """NetpyneInputDevice class to wrap around a NetPyNE input (stimulating) device"""

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "netpyne_input_device") #TODO: wrong model
        NetpyneDevice.__init__(self, device, netpyne_instance, *args, **kwargs)
        super(NetpyneInputDevice, self).__init__(device, netpyne_instance, *args, **kwargs)

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to."""
        return self.netpyne_instance.neuronsConnectedWith(self.label)
    
    @property
    def spiking_simulator_module(self):
        return self.device

class NetpyneSpikeGenerator(NetpyneInputDevice):

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "spike_generator_placeholder")
        super(NetpyneSpikeGenerator, self).__init__(device, netpyne_instance, *args, **kwargs)

NetpyneSpikeInputDeviceDict = {
                            "spike_generator_placeholder": NetpyneSpikeGenerator
                            }
NetpyneCurrentInputDeviceDict = {} # TODO: to be populated

NetpyneInputDeviceDict = {}
NetpyneInputDeviceDict.update(NetpyneSpikeInputDeviceDict)
NetpyneInputDeviceDict.update(NetpyneCurrentInputDeviceDict)

# Output devices

class NetpyneOutputDevice(NetpyneDevice):

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        NetpyneDevice.__init__(self, device, netpyne_instance, *args, **kwargs)
        # super(NetpyneOutputDevice, self).__init__(device, netpyne_instance, *args, **kwargs)
    
    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected from."""
        return self.netpyne_instance.cellGidsForPop(self.population_label)

        # neurons = []
        # for pop in self.spiking_populations_labels:
        #     neurons.append(
        #         self.device.neuronsConnecting(internalPopulation=pop)
        #     )
        # return numpy.array(neurons).flatten()

class NetpyneSpikeRecorder(NetpyneOutputDevice, SpikeRecorder):

    def __init__(self, device=None, netpyne_instance=None, **kwargs):
        # kwargs["model"] = kwargs.get("model", "spike_recorder")
        NetpyneOutputDevice.__init__(self, device, netpyne_instance, **kwargs)
        SpikeRecorder.__init__(self, device, **kwargs)
    
    @property
    def events(self):
        return [] # TODO: never called, with 'rate' mode at least

    @property
    def number_of_events(self):
        # TODO: now it returns new events, not total 
        return self.device.numberOfSpikes(self.population_label)

    def reset(self):
        # self._reset()
        pass

    def get_new_events(self, variables=None, **filter_kwargs):
        pass

    @property
    def new_events(self):
        return self.get_new_events()

    @property
    def number_of_new_events(self):
        """This method returns the number (integer) of events"""
        return self.device.numberOfSpikes(self.population_label)

    @property
    def spiking_simulator_module(self):
        return self.device.netpyne_instance

NetpyneOutputSpikeDeviceDict = {"spike_recorder": NetpyneSpikeRecorder}
NetpyneOutputContinuousTimeDeviceDict = {} # TODO: to be populated


NetpyneOutputDeviceDict = {}
NetpyneOutputDeviceDict.update(NetpyneOutputSpikeDeviceDict)
NetpyneOutputDeviceDict.update(NetpyneOutputContinuousTimeDeviceDict)