from tvb_multiscale.core.spiking_models.region_node import SpikingRegionNode


class NetpyneRegionNode(SpikingRegionNode):

    """NetpyneRegionNode class is an indexed mapping
       (based on inheriting from pandas.Series class)
       between populations labels and netpyne.NodeCollection instances,
       residing at a specific brain region node.
    """

    netpyne_instance = None
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, label="", input_nodes=None, netpyne_instance=None, **kwargs):
        self.netpyne_instance = netpyne_instance
        super(NetpyneRegionNode, self).__init__(label, input_nodes, **kwargs)

    @property
    def spiking_simulator_module(self):
        if self.netpyne_instance is None:
            for i_pop, pop_lbl, pop in self._loop_generator():
                self.netpyne_instance = pop.netpyne_instance
                if self.netpyne_instance is not None:
                    break
        return self.netpyne_instance