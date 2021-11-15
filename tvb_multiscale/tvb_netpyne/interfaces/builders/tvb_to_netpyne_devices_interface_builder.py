from tvb_multiscale.tvb_netpyne.interfaces.tvb_to_netpyne_devices_interface import INPUT_INTERFACES_DICT
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import create_device, connect_device
from tvb_multiscale.core.interfaces.builders.tvb_to_spikeNet_device_interface_builder import \
    TVBtoSpikeNetDeviceInterfaceBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices


class TVBtoNetpyneDeviceInterfaceBuilder(TVBtoSpikeNetDeviceInterfaceBuilder):
    _available_input_device_interfaces = INPUT_INTERFACES_DICT

    @property
    def netpyne_instance(self):
        return self.spiking_network.netpyne_instance

    @property
    def spiking_dt(self):
        try:
            return self.netpyne_instance.dt()
        except:
            return super(TVBtoNetpyneDeviceInterfaceBuilder, self).spiking_dt

    @property
    def min_delay(self):
        try:
            return self.netpyne_instance.minDelay()
        except:
            return self.default_min_delay

    def build_and_connect_devices(self, devices, nodes, *args, **kwargs):
        return build_and_connect_devices(devices, create_device, connect_device,
                                         nodes, self.config, netpyne_instance=self.netpyne_instance)