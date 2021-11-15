from tvb_multiscale.core.spiking_models.brain import SpikingBrain

# TODO: why is this ever needed?
class NetpyneBrain(SpikingBrain):

    """"NetpyneBrain is an indexed mapping (based on inheriting from pandas.Series class)
       between brain regions' labels and
       the respective NetpyneRegionNode instances.
    """

    netpyne_instance = None
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, input_brain=None, netpyne_instance=None, **kwargs):
        self.netpyne_instance = netpyne_instance
        super(NetpyneBrain, self).__init__(input_brain, **kwargs)

    # def get_number_of_neurons(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None):
    #     """Method to get the number of neurons of the SpikingBrain.
    #        Argument:
    #         reg_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected regions.
    #                           Default = None, corresponds to all regions of the SpikingBrain.
    #         pop_inds_or_lbls: collection (list, tuple, array) of the indices or keys of selected populations.
    #                           Default = None, corresponds to all populations of each SpikingRegionNode.
    #        Returns:
    #         int: number of neurons.
    #     """
    #     return 0# len(self.get_neurons(reg_inds_or_lbls, pop_inds_or_lbls))

    @property
    def spiking_simulator_module(self):
        # TODO: seems to never be called
        if self.netpyne_instance is None:
            for i_pop, pop_lbl, pop in self._loop_generator():
                self.netpyne_instance = pop.netpyne_instance
                if self.netpyne_instance is not None:
                    break
        return self.netpyne_instance