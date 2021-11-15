import neuron
import numpy
from tvb_multiscale.core.spiking_models.devices import Device, InputDevice, OutputDevice, SpikeRecorder

class NetpyneDevice(Device):
    
    def _print_neurons(self):
        pass

    def _assert_device(self):
        """Method to assert that the node of the network is a device"""
        pass

    # Methods to get or set attributes for devices and/or their connections:

    def Set(self, values_dict):
        """Method to set attributes of the device
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        if self.model == "poisson_generator":
            raise NotImplementedError
        elif self.model == "spike_generator_placeholder":
            for pop in self.spiking_populations_labels:
                self.device.convertFiringRate(values_dict, self.label, pop)

    def Get(self, attrs=None):
        """Method to get attributes of the device.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all device's attributes.
           Returns:
            Dictionary of attributes.
        """
        pass

    def GetConnections(self):
        """Method to get connections of the device to/from neurons.
           Returns:
            connections' objects.
        """
        # TODO: confirm/deny that tvb implementation is interested only in number of connectins. If so, it's equivalent to number of artificial neurons/stimuli
        return self.neurons

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

    # @property
    # @abstractmethod
    # def connections(self):
    #     """Method to get all connections of the device to/from the device.
    #        Returns:
    #         connections' objects.
    #     """
    #     pass

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to/from."""
        pass # overriden in subclasses

class NetpyneInputDevice(NetpyneDevice, InputDevice):

    """NetpyneInputDevice class to wrap around a NetPyNE input (stimulating) device"""

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        self.spiking_populations_labels = []
        kwargs["model"] = kwargs.pop("model", "netpyne_input_device")
        super(NetpyneInputDevice, self).__init__(device, netpyne_instance, *args, **kwargs)

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to."""
        neurons = []
        for pop in self.spiking_populations_labels:
            neurons.append(
                self.device.neuronsConnecting(internalPopulation=pop, externalNode=self.label)
            )
        return numpy.array(neurons).flatten()

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

class NetpyneOutputDevice(NetpyneDevice, OutputDevice):

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        self.spiking_populations_labels = []
        super(NetpyneOutputDevice, self).__init__(device, netpyne_instance, *args, **kwargs)
    
    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected from."""
        neurons = []
        for pop in self.spiking_populations_labels:
            neurons.append(
                self.device.neuronsConnecting(internalPopulation=pop)
            )
        return numpy.array(neurons).flatten()

class NetpyneSpikeRecorder(NetpyneOutputDevice, SpikeRecorder):
    
    @property
    def events(self):
        return [] # TODO: never called, with 'rate' mode at least

    @property
    def number_of_events(self):
        # self.label is for ex. Rin_e_parahippocampal_L (so 'node' can be taken from here)
        # self.device is NetpyneProxyDevice so if we store node there..
        # also, self.neurons seems to have neurons gids
        return self.device.numberOfSpikes(self.population_label)

    def reset(self):
        # self._reset()
        pass

NetpyneOutputSpikeDeviceDict = {"spike_recorder": NetpyneSpikeRecorder}
NetpyneOutputContinuousTimeDeviceDict = {} # TODO: to be populated


NetpyneOutputDeviceDict = {}
NetpyneOutputDeviceDict.update(NetpyneOutputSpikeDeviceDict)
NetpyneOutputDeviceDict.update(NetpyneOutputContinuousTimeDeviceDict)