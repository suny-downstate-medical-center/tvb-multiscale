from tvb_multiscale.tvb_netpyne.config import CONFIGURED
from tvb_multiscale.tvb_netpyne.netpyne_models.devices import \
     NetpyneInputDeviceDict, NetpyneSpikeInputDeviceDict, NetpyneCurrentInputDeviceDict, NetpyneOutputDeviceDict, NetpyneOutputSpikeDeviceDict, NetpyneOutputContinuousTimeDeviceDict
from tvb_multiscale.core.interfaces.base import TVBSpikeNetInterface


class TVBNetpyneInterface(TVBSpikeNetInterface):
    # TODO: populate with proper stuff
    _available_input_devices = NetpyneInputDeviceDict.keys()
    _current_input_devices = NetpyneCurrentInputDeviceDict.keys()
    _spike_rate_input_devices = NetpyneSpikeInputDeviceDict.keys()
    _available_output_devices = NetpyneOutputDeviceDict.keys()
    _spike_rate_output_devices = NetpyneOutputSpikeDeviceDict.keys()
    _multimeter_output_devices = NetpyneOutputContinuousTimeDeviceDict.keys()
    _voltmeter_output_devices = ["voltmeter"]

    def __init__(self, config=CONFIGURED):
        super(TVBNetpyneInterface, self).__init__(config)

    @property
    def netpyne_instance(self):
        return self.spiking_network.netpyne_instance