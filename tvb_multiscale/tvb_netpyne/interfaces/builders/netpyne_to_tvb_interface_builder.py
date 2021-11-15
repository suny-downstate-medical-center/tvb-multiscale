from tvb_multiscale.tvb_netpyne.interfaces.netpyne_to_tvb_interface import NetpyneToTVBInterface
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import create_device, connect_device

from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices
from tvb_multiscale.core.interfaces.builders.spikeNet_to_tvb_interface_builder import SpikeNetToTVBInterfaceBuilder


class NetpyneToTVBInterfaceBuilder(SpikeNetToTVBInterfaceBuilder):
    _build_target_class = NetpyneToTVBInterface

    @property
    def netpyne_instance(self):
        return self.spiking_network.netpyne_instance

    def build_and_connect_devices(self, devices, nodes, *args, **kwargs):
        return build_and_connect_devices(devices, create_device, connect_device,
                                         nodes, self.config, netpyne_instance=self.netpyne_instance)