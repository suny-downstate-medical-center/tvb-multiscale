from random import randint, uniform
# from ANNarchy.core.Simulate import simulate
import netpyne
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.utils import source

from netpyne import specs, sim
from netpyne.sim import *

class NetpyneCellGeometry(object):

    def __init__(self, diam, length, axialResistance) :
        self.diam = diam
        self.length = length
        self.axialR = axialResistance

    def toDict(self):
        return {'diam': self.diam, 'L': self.length, 'Ra': self.axialR}

class NetpyneMechanism(object):

    def __init__(self, name, gNaBar, gKBar, gLeak, eLeak):
        self.name = name
        self.gNaBar = gNaBar
        self.gKBar = gKBar
        self.gLeak = gLeak
        self.eLeak = eLeak

    def toDict(self):
        return {'gnabar': self.gNaBar, 'gkbar': self.gKBar, 'gl': self.gLeak, 'el': self.eLeak}

class NetpyneCellModel(object):

    def __init__(self, name, geom, mech):
        self.name = name
        self.geom = geom
        self.mech = mech

    def geometry(self):
        return self.geom.toDict()

    def getMech(self):
        return self.mech.name, self.mech.toDict()


class NetpyneInstance(object):
    
    def __init__(self, dt, simDurationFunc):

        self.dt = dt
        self.simDurationFunc = simDurationFunc

        self.spikingPopulationLabels = []

        self.netParams = specs.NetParams()

        # using VecStim model from NEURON for artificial cells serving as stimuli
        self.netParams.cellParams['art_NetStim'] = {'cellModel': 'DynamicNetStim'}

        ## Synaptic mechanism parameters
        # hia: de-hardcode 'em
        self.netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.8, 'tau2': 5.3, 'e': 0}  # NMDA
        self.netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': 0.6, 'tau2': 8.5, 'e': -75}  # GABA

        # Simulation options
        simConfig = specs.SimConfig()

        simConfig.dt = dt
        # simConfig.verbose = True

        simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
        simConfig.analysis['plotTraces'] =  {'include': [('parahippocampal_L.E', [0, 10, 20, 30, 40]), ('parahippocampal_L.I', [0, 10, 20, 30, 40])], 'saveFig': True}
        simConfig.analysis['plotRaster'] = {'include': ['parahippocampal_L.E', 'parahippocampal_L.I'], 'saveFig': True} 
        
        simConfig.recordStep = np.min(1.0, dt * 10)
        simConfig.savePickle = False        # Save params, network and sim output to pickle file
        simConfig.saveJson = False

        self.simConfig = simConfig

    def registerCellModel(self, cellModel):
        cellParams = netpyne.specs.Dict()

        cellParams.secs.soma.geom = cellModel.geom.toDict()
        #  {'diam': 18.8, 'L': 18.8, 'Ra': 123.0}
        mechName, mech = cellModel.getMech()
        cellParams.secs.soma.mechs[mechName] = mech
        

        self.netParams.cellParams[cellModel.name] = cellParams

    def minDelay(self):
        return self.dt

    def createCells(self):
        self.simConfig.recordCellsSpikes = self.spikingPopulationLabels # to exclude stimuli-cells

        sim.initialize(self.netParams, self.simConfig)  # create network object and set cfg and net params
        sim.net.createPops()                  # instantiate network populations
        sim.net.createCells()                 # instantiate network cells based on defined populations
    
    def createAndPrepareNetwork(self): # TODO: bad name?
        sim.cfg.duration = self.simDurationFunc()

        sim.setNetParams(self.netParams)

        sim.net.connectCells()
        sim.net.addStims()
        sim.net.addRxD()
        sim.setupRecording()

        prepareContinuousRun()

    def connectStimuli(self, sourcePop, targetPop, weight, delay, receptorType, scale):

        # first create artificial cells serving as stimulus, one per each target cell

        sourceCells = self.netParams.popParams[sourcePop]['numCells']
        targetCells = self.netParams.popParams[targetPop]['numCells']

        # one-to-one connection, scaled by 'lamda' of connectivity (TVB-defined scale)
        prob = 1.0 / sourceCells
        prob *= scale

        connLabel = sourcePop + '->' + targetPop
        self.netParams.connParams[connLabel] = {
            'preConds': {'pop': sourcePop},
            'postConds': {'pop': targetPop},
            'probability': prob,
            'weight': weight,
            'delay': delay,
            'synMech': receptorType
        }

    def interconnectSpikingPopulations(self, sourcePopulation, targetPopulation, synapticMechanism, weight, delay, probabilityOfConn):

        delayFunc = 'max(1, normal(' + str(delay) + ',2))'

        label = sourcePopulation + "->" + targetPopulation
        self.netParams.connParams[label] = {
            'preConds': {'pop': sourcePopulation},       # conditions of presyn cells
            'postConds': {'pop': targetPopulation},      # conditions of postsyn cells
            'probability': probabilityOfConn,
            'weight': weight,
            'delay': delayFunc,
            'synMech': synapticMechanism }

    def registerPopulation(self, label, cellModel, size):
        self.spikingPopulationLabels.append(label)
        self.netParams.popParams[label] = {'cellType': cellModel, 'numCells': size}

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

        t = h.t # TODO: here and in other places: don't use h.t directly, but through netpyne engine

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

        if rate == 0.0:
            return

        stimulusCellGids = sim.net.pops[sourcePop].cellGids

        # TODO.TVB: some crazy values of rate are conveyed in the beginning of simulation. Is this expected?

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
        timewind = 0.1 # TODO: de-hardcode time

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
        # TODO: make sure this is always called
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
        allSpikes = poisson_generator(rate, 0, totalDuration, 5)

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
