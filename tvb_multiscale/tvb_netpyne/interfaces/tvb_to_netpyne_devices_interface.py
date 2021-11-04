import numpy as np
from tvb_multiscale.core.interfaces.tvb_to_spikeNet_device_interface import TVBtoSpikeNetDeviceInterface

# Each interface has its own set(values) method, depending on the underlying device:

class TVBtoNetpyneDeviceInterface(TVBtoSpikeNetDeviceInterface):

    @property
    def netpyne_instance(self):
        return self.spiking_network.netpyne_instance

class TVBtoNetpynePoissonGeneratorInterface(TVBtoNetpyneDeviceInterface):

    def set(self, values):
        self.Set({"rates": np.maximum([0], self._assert_input_size(values))})

    # def set(self, values):
    #     self.Set({"rate": np.maximum([0], self._assert_input_size(values)),
    #               "origin": self.nest_instance.GetKernelStatus("time"),
    #               "start": self.nest_instance.GetKernelStatus("min_delay"),
    #               "stop": self.dt})

class TVBtoNetpyneSpikeGeneratorInterface(TVBtoNetpyneDeviceInterface):

     def set(self, values):
        self.Set({"rates": np.maximum([0], self._assert_input_size(values))})
#     def set(self, values):
#         values = self._assert_input_size(values)
#         # TODO: change this so that rate corresponds to number of spikes instead of spikes' weights
#         self.Set({"spikes_times": np.ones((self.number_of_nodes,)) *
#                                   0,#self.nest_instance.GetKernelStatus("min_delay"),
#                   "origin": 0,#self.nest_instance.GetKernelStatus("time"),
#                   "spike_weights": values})

INPUT_INTERFACES_DICT = {
                         "spike_generator_placeholder": TVBtoNetpyneSpikeGeneratorInterface,
                         "poisson_generator": TVBtoNetpynePoissonGeneratorInterface
                         }