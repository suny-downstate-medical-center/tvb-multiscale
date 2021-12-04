import netpyne
import numpy as np

from tvb_multiscale.core.interfaces.spikeNet_to_tvb_interface import SpikeNetToTVBinterface

class NetpyneToTVBInterface(SpikeNetToTVBinterface):

    def __init__(self, spiking_network, tvb_sv_id, name="", model="",
                 nodes_ids=[], scale=np.array([1.0]), device_set=None):
        super(NetpyneToTVBInterface, self).__init__(spiking_network, tvb_sv_id, name, model,
                                                 nodes_ids, scale, device_set)
        self.number_of_events = np.zeros((len(self.nodes_ids),))
    
    @property
    def netpyne_instance(self):
        return self.spiking_network.netpyne_instance

    @property
    def population_mean_spikes_number(self):
        
        values = []
        spktimes, spkids = self.netpyne_instance.latestSpikes(self.tvb_dt)

        if len(spktimes) == 0:
            return np.zeros((len(self.nodes_ids),))
        nodes = self.devices()
        for node in nodes:
            device = self[node]
            cellGidsForNode = self.netpyne_instance.cellGidsForPop(device.population_label)
            inNode = np.isin(spkids, cellGidsForNode)
            numSpikes = len(spkids[inNode])
            if numSpikes > 0:
                num = numSpikes / self[node].number_of_neurons
                print(f"Netpyne:: recorded {num} spikes per neuron from {device.label}. Approx rate =  {num * 1000 / self.tvb_dt}")
                values.append(num)
            else:
                values.append(0.0)
        return np.array(values).flatten()

    # @property
    # def population_mean_spikes_activity(self):
    #     values = []
    #     for i_node, (node, n_events) in enumerate(zip(self.devices(), self.number_of_events)):
    #         number_of_events = self[node].number_of_events
    #         if number_of_events > n_events:
    #             # if there are new events, sum up only the spikes' weights not yet considered:
    #             values.append(np.sum(self.events[self.spike_var][n_events:]) / self[node].number_of_neurons)
    #         else:
    #             values.append(0.0)
    #         self.number_of_events[i_node] = number_of_events
    #     return np.array(values).flatten()

    @property
    def current_population_mean_values(self):
        # TODO: used only with multimeter and voltmeter
        values = []
        for i_node, (node, n_events) in enumerate(zip(self.devices(), self.number_of_events)):
            number_of_events = self[node].number_of_events
            if number_of_events > n_events:
                # if there are new events, take the mean of all the values not yet considered:
                # (unlike discrete spike events, for continuous multimeter events
                # we assume that the new events are of size n_time_steps * n_neurons,
                # and, unlike spike weights' time series,
                # we compute mean absolute activity and not a rate
                # (i.e., with division by number of time points instead of time)
                this_node_values = []
                for var in self[node].record_from:
                    this_node_values.append(np.mean(self.events[var][n_events:]))
                values.append(np.array(this_node_values))
            else:
                values.append(np.zeros((len(self.record_from),)))
            self.number_of_events[i_node] = number_of_events
        return np.array(values).flatten()