import numpy as np
from tvb_multiscale.core.interfaces.tvb_to_spikeNet_device_interface import TVBtoSpikeNetDeviceInterface

# Each interface has its own set(values) method, depending on the underlying device:

class TVBtoNetpyneDeviceInterface(TVBtoSpikeNetDeviceInterface):

    @property
    def netpyne_instance(self):
        return self.spiking_network.netpyne_instance

class TVBtoNetpynePoissonGeneratorInterface(TVBtoNetpyneDeviceInterface):

    def set(self, values):
        # TODO: sanity-check rate here?
        self.Set({"rates": np.maximum([0], self._assert_input_size(values)),
                  "dt": self.dt                
                })

INPUT_INTERFACES_DICT = {
                         "poisson_generator": TVBtoNetpynePoissonGeneratorInterface
                         }