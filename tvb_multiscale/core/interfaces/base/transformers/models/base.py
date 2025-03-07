# -*- coding: utf-8 -*-

from copy import deepcopy
from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import Float, NArray, Attr

from tvb_multiscale.core.neotraits import HasTraits


class Transformer(HasTraits):
    __metaclass__ = ABCMeta

    """
        Abstract Transformer base class comprising:
            - an input buffer data,
            - an output buffer data,
            - an abstract method for the computations applied 
              upon the input buffer data for the output buffer data to result.
    """

    input_buffer = []

    output_buffer = []

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    input_time = NArray(
        dtype=np.int,
        label="Input time vector",
        doc="""Buffer of time (float) or time steps (integer) corresponding to the input buffer.""",
        required=True,
        default=np.array([]).astype("i")
    )

    output_time = NArray(
        dtype=np.int,
        label="Output time vector",
        doc="""Buffer of time (float) or time steps (integer) corresponding to the output bufer.""",
        required=True,
        default=np.array([]).astype("i")
    )

    time_shift = Float(
        label="Time shit",
        doc="""Time shift for output time/spike times, depending on the transformation. Default = 0.0""",
        required=True,
        default=0.0
    )

    def compute_time(self):
        self.output_time = np.copy(self.input_time) + np.round(self.time_shift / self.dt).astype("i")

    @abstractmethod
    def _compute(self, input_buffer, *args, **kwargs):
        """Abstract method for the computation on the input buffer data for the output buffer data to result.
           It returns the output of its computation"""
        pass

    def compute(self, *args, **kwargs):
        """Abstract method for the computation on the input buffer data for the output buffer data to result.
           It sets the output buffer property"""
        self.output_buffer = np.array(self._compute(self.input_buffer, *args, **kwargs))

    def __call__(self, data=None, time=None):
        if data is not None:
            self.input_buffer = data
        if time is not None:
            self.input_time = time
        self.compute_time()
        self.compute()
        return [self.output_time, self.output_buffer]

    def __getattribute__(self, attr):
        return super().__getattribute__(attr)

    def __setattr__(self, attr, val):
        return super().__setattr__(attr, val)

    def _assert_size(self, attr, buffer="input", dim=0):
        value = getattr(self, attr)
        buffer_size = len(getattr(self, "%s_buffer" % buffer))
        if buffer_size != 0:
            if buffer_size != value.shape[dim]:
                if value.shape[dim] == 1:
                    value = np.repeat(value, buffer_size, axis=dim)
                else:
                    raise ValueError("%s (=%s) is neither of length 1 "
                                     "nor of length equal to the proxy dimension (%d) of the buffer (=%d)"
                                     % (attr, str(value), dim, buffer_size))
                setattr(self, attr, value)
        return value

    def info(self, recursive=0):
        info = super(Transformer, self).info(recursive=recursive)
        keys = list(info.keys())
        for buffer in ["input_buffer", "output_buffer"]:
            if buffer not in keys:
                info[buffer] = getattr(self, buffer)
        return info


# A few basic examples:


class Elementary(Transformer):
    """
        Elementary Transformer just copies the input to the output without any computation.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a method to copy the input buffer data to the output buffer.
    """

    def _compute(self, input_buffer):
        """Method that just returns the input buffer data."""
        return deepcopy(input_buffer)


class Linear(Transformer):
    """
        Linear Transformer scales the input with a scale factor and translates it by a constant
        in order to compute the output.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a scale factor numpy.array,
            - a translation factor numpy.array,
            - a method to multiply the input buffer data by the scale factor for the output buffer data to result.
    """

    scale_factor = NArray(
        label="Scale factor",
        doc="""Array to scale input buffer.""",
        required=True,
        default=np.array([1.0])
    )

    translation_factor = NArray(
        label="Translation factor",
        doc="""Array to translate input buffer.""",
        required=True,
        default=np.array([0.0])
    )

    @property
    def _scale_factor(self):
        return self._assert_size("scale_factor")

    @property
    def _translation_factor(self):
        return self._assert_size("translation_factor")

    def _compute(self, input_buffer):
        """Method that just scales and translates input buffer data to compute the output buffer data."""
        output_buffer = []
        for proxy_buffer, scale_factor, translation_factor in \
                zip(input_buffer, self._scale_factor, self._translation_factor):
            output_buffer.append(scale_factor * proxy_buffer + translation_factor)
        return output_buffer


class LinearRate(Linear):

    """LinearRate class that just scales and translates mean field rates,
       including any unit conversions and conversions from mean field to total rates"""

    pass


class LinearCurrent(Linear):
    """LinearCurrent class that just scales and translates mean field currents,
       including any unit conversions and conversions from mean field to total rates"""

    pass


class LinearPotential(Linear):
    """LinearPotential class that just scales and translates mean field membrane potentials
       including any unit conversions and conversions from mean field to total rates"""

    pass


class RatesToSpikes(LinearRate):
    __metaclass__ = ABCMeta

    """
        RatesToSpikes Transformer abstract base class
    """

    input_buffer = NArray(
        label="Input buffer",
        doc="""Array to store temporarily the data to be transformed.""",
        required=True,
        default=np.array([])
    )

    output_buffer = Attr(
        field_type=np.ndarray,
        doc="""Array of spiketrains storing temporarily the generated spikes.""",
        default=np.array([])
    )

    number_of_neurons = NArray(
        dtype="i",
        label="Number of neurons",
        doc="""Number of neuronal spiketrains to generate for each proxy node.""",
        required=True,
        default=np.array([1]).astype('i')
    )

    @property
    def _number_of_neurons(self):
        return self._assert_size("number_of_neurons")

    @property
    def _t_start(self):
        return (self.dt * (self.input_time[0] - 1) + self.time_shift) * self.ms

    @property
    def _t_stop(self):
        return (self.dt * self.input_time[-1] + self.time_shift) * self.ms

    @abstractmethod
    def _compute_spiketrains(self, rates, proxy_count, *args, **kwargs):
        """Abstract method for the computation of rates data transformation to spike trains."""
        pass

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        rates = self.scale_factor * input_buffer + self.translation_factor
        output_buffer = []
        for iP, proxy_rate in enumerate(rates):
            output_buffer.append(self._compute_spiketrains(proxy_rate, iP, *args, **kwargs))
        return output_buffer


class SpikesToRates(LinearRate):
    __metaclass__ = ABCMeta

    """
        RateToSpikes Transformer abstract base class
    """

    input_buffer = Attr(
        field_type=np.ndarray,
        doc="""Array of spiketrains (lists) storing temporarily the spikes to be transformed into rates.""",
        default=np.array([])
    )

    output_buffer = NArray(
        label="Output buffer",
        doc="""Array to store temporarily the output rate data.""",
        required=True,
        default=np.array([])
    )

    @property
    def _t_start(self):
        return (self.dt * (self.input_time[0] - 1) + self.time_shift) * self.ms

    @property
    def _t_stop(self):
        return (self.dt * self.input_time[-1] + self.time_shift) * self.ms

    @abstractmethod
    def _compute_rates(self, spikes, *args, **kwargs):
        """Abstract method for the computation of spike trains data transformation
           to instantaneous mean spiking rates."""
        pass

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        output_buffer = []
        for proxy_buffer, scale_factor, translation_factor in \
                zip(input_buffer, self._scale_factor, self._translation_factor):
            # At this point we assume that input_buffer has shape (proxy,)
            output_buffer.append(scale_factor * self._compute_rates(proxy_buffer, *args, **kwargs) + translation_factor)
        return output_buffer
