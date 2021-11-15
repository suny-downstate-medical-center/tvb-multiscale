from tvb_multiscale.core.config import Config
from tvb_multiscale.core.utils.log_utils import initialize_logger as initialize_logger_base

class NetpyneConfig(Config):
    
    DEFAULT_MODEL = "default_model_placeholder" # TODO: what's the default model?

    # Delays should be at least equal to NetPyNE time resolution
    DEFAULT_CONNECTION = {"synapse_model": "synapse_model_placeholder", "weight": 1.0, "delay": 1.0, 'receptor_type': 0,
                          "source_inds": None, "target_inds": None, "params": {},
                          "conn_spec": {"allow_autapses": True, 'allow_multapses': True, 'rule': "all_to_all",
                                        "indegree": None, "outdegree": None, "N": None, "p": 0.1}}

    NETPYNE_INPUT_DEVICES_PARAMS_DEF = {"poisson_generator": {"allow_offgrid_times": False},
                                        "spike_generator_placeholder": {"allow_offgrid_times": False}
                                        }
    NETPYNE_OUTPUT_DEVICES_PARAMS_DEF = {
                                        #  "multimeter": {"record_from": ["V_m"], "record_to": "memory"},
                                        #  "voltmeter": {"record_to": "memory"},
                                         "spike_recorder": {"record_to": "memory"},
                                        #  "spike_multimeter": {'record_from': ["spike"], "record_to": "memory"}
                                         }

CONFIGURED = NetpyneConfig(initialize_logger=False)

def initialize_logger(name, target_folder=None):
    if target_folder is None:
        target_folder = Config().out.FOLDER_LOGS
    return initialize_logger_base(name, target_folder)