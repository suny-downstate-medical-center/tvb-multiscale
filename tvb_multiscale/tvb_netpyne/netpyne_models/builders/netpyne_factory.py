from copy import deepcopy

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error

from tvb_multiscale.tvb_netpyne.config import CONFIGURED
from tvb_multiscale.tvb_netpyne.netpyne_models.devices import NetpyneInputDeviceDict, NetpyneOutputDeviceDict
from tvb_multiscale.tvb_netpyne.ext.instance import NetpyneInstance

def create_device(device_model, params={}, config=CONFIGURED, netpyne_instance=None, **kwargs):
    """Method to create a NetpynDevice.
       Arguments:
        device_model: name (string) of the device model
        params: dictionary of parameters of device and/or its synapse. Default = None
        config: configuration class instance. Default: imported default CONFIGURED object.
        netpyne_instance: the NetPyNE instance. Default = None, in which case we are going to load one, and also return it in the output
       Returns:
        the NetpyneDevice class, and optionally, the NetPyNE instance if it is loaded here.
    """
    
    # Get the default parameters for this device...
    if device_model in NetpyneInputDeviceDict.keys():
        devices_dict = NetpyneInputDeviceDict
        default_params = deepcopy(config.NETPYNE_INPUT_DEVICES_PARAMS_DEF.get(device_model, {}))
    elif device_model in NetpyneOutputDeviceDict.keys():
        devices_dict = NetpyneOutputDeviceDict
        default_params = deepcopy(config.NETPYNE_OUTPUT_DEVICES_PARAMS_DEF.get(device_model, {}))
    else:
        raise_value_error("%s is neither one of the available input devices: %s\n "
                          "nor of the output ones: %s!" %
                          (device_model, str(config.NETPYNE_INPUT_DEVICES_PARAMS_DEF),
                           str(config.NETPYNE_OUTPUT_DEVICES_PARAMS_DEF)))

    # ...and update them with any user provided parameters
    label = kwargs.pop("label", "")
    default_params["label"] = label
    if isinstance(params, dict) and len(params) > 0:
        default_params.update(params)

    netpyne_device = netpyne_instance.createDevice(device_model, params=default_params)
    
    DeviceClass = devices_dict[device_model]
    device = DeviceClass(netpyne_device, netpyne_instance=netpyne_instance, label=label)
    device.model = device_model # TODO: nest passes this through initializers chain and assigns deeply in `_NESTNodeCollection` or so
    device.label = label
    return device


def connect_device(netpyne_device, population, neurons_inds_fun, weight=1.0, delay=0.0, receptor_type=0,
                   syn_spec=None, conn_spec=None, config=CONFIGURED, **kwargs):
    """This method connects a NetpyneDevice to a NetpynePopulation instance.
       Arguments:
        netpyne_device: the NetpyneDevice instance
        population: the NetpynePopulation instance
        neurons_inds_fun: a function to return a NetpynePopulation or a subset thereof of the target population.
                          Default = None.
        weight: the weights of the connection. Default = 1.0.
        delay: the delays of the connection. Default = 0.0.
        receptor_type: type of the synaptic receptor. Default = 0.
        config: configuration class instance. Default: imported default CONFIGURED object.
       Returns:
        the connected NetpyneDevice
    """

    netpyne_instance = netpyne_device.netpyne_instance
    spiking_population_label = population.brain_region + "." + population.label # TODO: factor out this label creation
    if netpyne_device.model in config.NETPYNE_INPUT_DEVICES_PARAMS_DEF:
        
        target = spiking_population_label
        print(f"Netpyne:: will connect input device {netpyne_device.model}. {netpyne_device.label} -> {target}")
        # TODO: process `receptor_type` somehow?
        netpyne_instance.createExternalConnection(sourcePop=netpyne_device.label, targetPop=target, weight=weight, delay=delay)
    elif netpyne_device.model in config.NETPYNE_OUTPUT_DEVICES_PARAMS_DEF:
        
        netpyne_device.population_label = spiking_population_label
        print(f"Netpyne:: will connect output device {netpyne_device.model} -- {netpyne_device.population_label}")
    else:
        print(f"Netpyne:: couldn't connect device. Unknown model {netpyne_device.model}")

    return netpyne_device