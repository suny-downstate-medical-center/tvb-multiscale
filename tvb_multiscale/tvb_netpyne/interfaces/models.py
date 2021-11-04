from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI

from tvb_multiscale.tvb_netpyne.interfaces.base import TVBNetpyneInterface
from ..config import CONFIGURED

class RedWWexcIOinhI(TVBNetpyneInterface):
    tvb_model = ReducedWongWangExcIOInhI()

    def __init__(self, config=CONFIGURED):
        super(RedWWexcIOinhI, self).__init__(config)
        # LOG.info("%s created!" % self.__class__)