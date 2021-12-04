from random import randint, uniform
# from ANNarchy.core.Simulate import simulate
import netpyne
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.utils import source
from tvb_multiscale.tvb_netpyne.ext.NodeCollection import NodeCollection

from netpyne import specs, sim
from netpyne.sim import *

# TODO: NetpyneInstance is stub
class NetpyneInstance(object):
    
    def __init__(self):
        self.spikingPopulationLabels = []

        self.netParams = specs.NetParams()
        cellParams = netpyne.specs.Dict()
        cellParams.secs.soma.geom = {'diam': 18.8, 'L': 18.8, 'Ra': 123.0}
        cellParams.secs.soma.mechs.hh = {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}

        self.netParams.cellParams['PYR'] = cellParams

        # using VecStim model from NEURON for artificial cells serving as stimuli
        self.netParams.cellParams['art_NetStim'] = {'cellModel': 'DynamicNetStim'}

        ## Synaptic mechanism parameters
        self.netParams.synMechParams['exc'] = {
            'mod': 'Exp2Syn', 
            'tau1': 0.1, 
            'tau2': 1.0, 
            'e': 0} # excitatory synaptic mechanism

        self.netParams.synMechParams['inh'] = {
            'mod': 'Exp2Syn',
            'tau1': 0.1,
            'tau2': 1.0,
            'e': -70} # inhibitory synaptic mechanism

        # Simulation options
        simConfig = specs.SimConfig()

        simConfig.duration = 398 # ms # TODO: should be same as TVB duration
        simConfig.dt = self.dt()
        # simConfig.verbose = True

        # simConfig.recordCells = ['uE']

        simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
        simConfig.analysis['plotTraces'] =  {'include': [('parahippocampal_L.E', [0]), ('parahippocampal_L.I', [0])], 'saveFig': True}
        simConfig.recordStep = 0.05          # Step size in ms to save data (eg. V traces, LFP, etc)
        simConfig.savePickle = False        # Save params, network and sim output to pickle file
        simConfig.saveJson = False

        self.simConfig = simConfig

        ## Population parameters
        
        # netParams.popParams['I'] = {'cellType': 'PYR', 'numCells': 20}

        # simConfig = specs.SimConfig()

        # sim.create(netParams = netParams, simConfig = simConfig)

    def dt(self):
        # integration dt in milliseconds
        return 0.025 # TODO: should be decreased?

    def minDelay():
        return 0.025 # TODO

    def createCells(self):
        self.simConfig.recordCellsSpikes = self.spikingPopulationLabels # to exclude stimuli-cells

        sim.initialize(self.netParams, self.simConfig)  # create network object and set cfg and net params
        sim.net.createPops()                  # instantiate network populations
        sim.net.createCells()                 # instantiate network cells based on defined populations
    
    def createAndPrepareNetwork(self): # TODO: bad name?

        sim.setNetParams(self.netParams)

        sim.net.connectCells()
        sim.net.addStims()
        sim.net.addRxD()
        sim.setupRecording()

        prepareContinuousRun()

    def createExternalConnection(self, sourcePop, targetPop, weight, delay):

        # TODO: weight by default is 1.0 which is huge. Trace down where it comes from
        weight = 0.01

        # first create artificial cells serving as stimulus, one per each target cell
        
        # TODO: would be great to bring it back to work:
        # number = self.netParams.popParams[targetPop]['numCells']
        # self.createArtificialCells(sourcePop, number)

        connLabel = sourcePop + '->' + targetPop
        self.netParams.connParams[connLabel] = {
            'preConds': {'pop': sourcePop},
            'postConds': {'pop': targetPop},
            'convergence': 1, # TODO: ?
            'weight': weight,
            'delay': delay,
            'synMech': 'exc' # TODO: can (or should) this be de-hardcoded?
        }

    def createInternalConnection(self, sourcePopulation, targetPopulation, synapticMechanism, weight, delay):
        label = sourcePopulation + "->" + targetPopulation
        self.netParams.connParams[label] = {
            'preConds': {'pop': sourcePopulation},       # conditions of presyn cells
            'postConds': {'pop': targetPopulation},      # conditions of postsyn cells
            'probability': 0.1, # TODO: will get value from conn_spec (rule=all_to_all, allow_autapses, allow multapses etc.)
            'weight': weight,
            'delay': delay,
            'synMech': synapticMechanism }

    def createNodeCollection(self, brainRegion, popLabel, cellModel, size, params=None):
        collection = NodeCollection(brainRegion, popLabel, size)
        print(f"Netpyne:: Creating population '{collection.label}' of {size} neurons of type '{cellModel}'.")
        self.spikingPopulationLabels.append(collection.label)

        self.netParams.popParams[collection.label] = {'cellType': cellModel, 'numCells': size}
        return collection

    from sys import float_info
    def createArtificialCells(self, label, number, interval=float_info.max, params=None):
        print(f"Netpyne:: Creating artif cells for node '{label}' of {number} neurons")
        self.netParams.popParams[label] = {
            'cellType': 'art_NetStim',
            'numCells': number,
            # 'spkTimes': [0]
            'interval': interval,
            'start': 0,
            'number': 2,
            'noise': 0.0
        }

    def createDevice(self, model, params):
        print(f"Netpyne:: will create internal device: {model} --- {params}")
        return NetpyneProxyDevice(netpyne_instance=self) 

    def latestSpikes(self, timeWind):

        t = h.t # TODO: were and in other places: don't use h.t directly, but through netpyne engine

        spktvec = np.array(sim.simData['spkt'])
        spkgids = np.array(sim.simData['spkid'])
        inds = np.nonzero(spktvec > (t - timeWind)) # filtered by time

        spktvec = spktvec[inds]
        spkgids = spkgids[inds]

        return spktvec, spkgids

    def allSpikes(self, cellGids):
        spktimes = np.array(sim.simData['spkt'])
        spkgids = np.array(sim.simData['spkid'])

        inPop = np.isin(spkgids, cellGids)

        spktimes = spktimes[inPop]
        spkgids = spkgids[inPop]
        return spktimes, spkgids

    #rate = list(values_dict.values())[0]
    def applyFiringRate(self, rate, sourcePop, dt):        

        stimulusCellGids = sim.net.pops[sourcePop].cellGids

        # the input rate is scaled by number of neurons in stimulated population. decreasing it back:
        # TODO.tvb: is this expected?
        rate /= len(stimulusCellGids)

        # TODO.TVB: some crazy values of rate are conveyed in the beginning of simulation. Is this expected?
        # if rate > 400:
        #     # For now - scale them down
        #     rate = rate / 2.0

        # print(f"Netpyne:: apply firing rate {rate}: {sourcePop}")

        spikesPerNeuron = generateSpikesForPopulation(len(stimulusCellGids), rate, dt)
        for index, spikes in spikesPerNeuron:
            cell = sim.net.cells[stimulusCellGids[index]]
            cell.hPointp.spike_now()
    
    def cellGidsForPop(self, popLabel):
        return sim.net.pops[popLabel].cellGids

    def neuronsConnectedWith(self, targetPop):
        gids = []
        for connection in self.netParams.connParams.keys():
            if connection.find(targetPop) >= 0:
                pop = self.netParams.connParams[connection]['postConds']['pop']
                gids.append(self.cellGidsForPop(pop))
        gids = np.array(gids).flatten()
        return gids

    def run(self, length):
        runForInterval(length)

from neuron import h
class NetpyneProxyDevice(object):

    def __len__(self):
        return 1 # TODO: at least add some explanatory comment

    netpyne_instance = None

    def __init__(self, netpyne_instance):
        self.netpyne_instance = netpyne_instance

    def numberOfSpikes(self, popLabel):
        # TODO: not getting called. is this expected?
        # use only node (proper one), exclude artif cells' spikes
        cellGids = sim.net.pops[popLabel].cellGids

        t = h.t
        timewind = 0.1

        spktvec = np.array(sim.simData['spkt'])
        spkgids = np.array(sim.simData['spkid'])

        timeFilter = (spktvec < t) * (spktvec > t - timewind)
        cellsFilter = np.isin(spkgids, cellGids)
        inds = np.nonzero(timeFilter * cellsFilter)

        spktvec = spktvec[inds]
        spkgids = spkgids[inds]

        num = len(spktvec)
        return num

# TODO: copy-pasted from internals of runSimWithIntervalFunc
def prepareContinuousRun():
    """
    Function for/to <short description of `netpyne.sim.run.runSimWithIntervalFunc`>

    Parameters
    ----------
    interval : <type>
        <Short description of interval>
        **Default:** *required*

    func : <type>
        <Short description of func>
        **Default:** *required*


    """

    # from .. import sim
    sim.pc.barrier()
    sim.timing('start', 'runTime')
    preRun()
    h.finitialize(float(sim.cfg.hParams['v_init']))

    if sim.rank == 0: print('\nNetpyne:: Preparing for interval run ...')

done = False
def runForInterval(interval):
    if round(h.t) < sim.cfg.duration:
        sim.pc.psolve(min(sim.cfg.duration, h.t+interval))
    else:
        global done
        if done:
            return
        done = True
        sim.pc.barrier() # Wait for all hosts to get to this point
        sim.timing('stop', 'runTime')
        if sim.rank==0:
            print(('  Done; run time = %0.2f s; real-time ratio: %0.2f.' %
                (sim.timingData['runTime'], sim.cfg.duration/1000/sim.timingData['runTime'])))
        sim.gatherData()
        sim.analyze()

def generateSpikesForPopulation(numNeurons, rate, dt):

        # instead of generating spike trains for time dt for each neuron,
        # generate one spike train for time dt*numNeurons and break it down between neurons
        totalDuration = numNeurons * dt
        allSpikes = poisson_generator(rate, 0, totalDuration)

        # now divide spike train between n=numNeurons bins, and adjust time of each spike so that start of bin is treated as absolute time

        binDuration = totalDuration / numNeurons
        binStartTimes = np.arange(0, totalDuration, binDuration)
    
        spikesPerNeuron = []
        for i, binStart in enumerate(binStartTimes):
            timeFilter = (allSpikes >= binStart) * (allSpikes < binStart + binDuration)
            inds = np.nonzero(timeFilter)
            spikesInBin = allSpikes[inds] - binStart
            if len(spikesInBin) > 0:
                spikesPerNeuron.append((i, spikesInBin))
        return spikesPerNeuron

def poisson_generator(rate, t_start=0.0, t_stop=1000.0, seed=None):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if 
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    Inputs:
    -------
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)
        array   - if True, a np array of sorted spikes is returned,
                    rather than a SpikeTrain object.

    Examples:
    --------
        >> gen.poisson_generator(50, 0, 1000)
        >> gen.poisson_generator(20, 5000, 10000, array=True)

    See also:
    --------
        inh_poisson_generator, inh_gamma_generator, inh_adaptingmarkov_generator
    """

    rng = np.random.RandomState(seed)

    #number = int((t_stop-t_start)/1000.0*2.0*rate)

    # less wasteful than double length method above
    n = (t_stop-t_start)/1000.0*rate
    number = np.ceil(n+3*np.sqrt(n))
    if number<100:
        number = min(5+np.ceil(2*n),100)

    if number > 0:
        isi = rng.exponential(1.0/rate, int(number))*1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes+=t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if i==len(spikes):
        # ISI buf overrun
        
        t_last = spikes[-1] + rng.exponential(1.0/rate, 1)[0]*1000.0

        while (t_last<t_stop):
            extra_spikes.append(t_last)
            t_last += rng.exponential(1.0/rate, 1)[0]*1000.0
        
        spikes = np.concatenate((spikes,extra_spikes))

    else:
        spikes = np.resize(spikes,(i,))

    return spikes
