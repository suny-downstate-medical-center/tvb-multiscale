from tvb_multiscale.tvb_netpyne.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_netpyne.interfaces.base import TVBNetpyneInterface

from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan as WilsonCowanSimModel


LOG = initialize_logger(__name__)


class Linear(TVBNetpyneInterface):
    tvb_model = Linear()

    def __init__(self, config=CONFIGURED):
        super(Linear, self).__init__(config)
        LOG.info("%s created!" % self.__class__)


class RedWWexcIO(TVBNetpyneInterface):
    tvb_model = ReducedWongWangExcIO()

    def __init__(self, config=CONFIGURED):
        super(RedWWexcIO, self).__init__(config)
        LOG.info("%s created!" % self.__class__)


class RedWWexcIOinhI(TVBNetpyneInterface):
    tvb_model = ReducedWongWangExcIOInhI()

    def __init__(self, config=CONFIGURED):
        super(RedWWexcIOinhI, self).__init__(config)
        LOG.info("%s created!" % self.__class__)


class WilsonCowan(TVBNetpyneInterface):
    tvb_model = WilsonCowanSimModel()

    def __init__(self, config=CONFIGURED):
        super(WilsonCowan, self).__init__(config)
        LOG.info("%s created!" % self.__class__)