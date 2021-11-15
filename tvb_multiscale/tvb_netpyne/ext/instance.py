from random import randint
from ANNarchy.core.Simulate import simulate
import netpyne
import numpy as np
from numpy.core.fromnumeric import shape
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
        self.netParams.cellParams['art_VecStim'] = {'cellModel': 'VecStim'}

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
        simConfig = specs.SimConfig()       # object of class SimConfig to store simulation configuration

        simConfig.duration = 2*1e2          # Duration of the simulation, in ms
        simConfig.dt = self.dt()            # Internal integration timestep to use
        # simConfig.verbose = True           # Show detailed messages

        # simConfig.recordCells = ['uE']

        # simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
        simConfig.recordStep = 0.01          # Step size in ms to save data (eg. V traces, LFP, etc)
        simConfig.savePickle = False        # Save params, network and sim output to pickle file
        simConfig.saveJson = False

        self.simConfig = simConfig

        ## Population parameters
        
        # netParams.popParams['I'] = {'cellType': 'PYR', 'numCells': 20}

        # simConfig = specs.SimConfig()

        # sim.create(netParams = netParams, simConfig = simConfig)

    def dt(self):
        # integration dt in milliseconds
        return 0.05 # TODO: should be decreased?

    def minDelay():
        return 0.05 # TODO

    def createCells(self):
        self.simConfig.recordCellsSpikes = self.spikingPopulationLabels

        sim.initialize(self.netParams, self.simConfig)  # create network object and set cfg and net params
        sim.net.createPops()                  # instantiate network populations
        sim.net.createCells()                 # instantiate network cells based on defined populations
    
    def createAndPrepareNetwork(self): # TODO: bad name?
        sim.setNetParams(self.netParams)

        sim.net.connectCells()                # create connections between cells based on params
        sim.net.addStims()                    # add external stimulation to cells (IClamps etc)
        sim.net.addRxD()                    # add reaction-diffusion (RxD)
        sim.setupRecording()  

        # def intfun(time):
        #     print(f"INTFUN {time}")
        #     print(np.array(sim.simData['spkt']))
        prepareContinuousRun()

    def createExternalConnection(self, sourcePop, targetPop, weight, delay):

        sourcePop = sourcePop + ":" + targetPop
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
            'convergence': 1, # TODO: will get value from conn_spec (rule=all_to_all, allow_autapses, allow multapses etc.)
            'weight': weight,
            'delay': delay,
            'synMech': synapticMechanism }

    def createNodeCollection(self, nodeLabel, popLabel, cellModel, number, params):
        print(f"Netpyne:: Creating population in node {nodeLabel} labeled '{popLabel}' of {number} neurons of type '{cellModel}'.")
        label = nodeLabel + "." + popLabel
        self.spikingPopulationLabels.append(label)
        self.netParams.popParams[label] = {'cellType': cellModel, 'numCells': number}
        return NodeCollection(label, cellModel, number, params)

    from random import randint
    def createArtificialCells(self, tvbStateVar, nodeLabel, projectToNode, projectToPop, number, params=None):
        print(f"Netpyne:: Creating artif cells for node {nodeLabel} of {number} neurons.")
        targetCellsLabel = projectToNode + "." + projectToPop
        label = tvbStateVar + "_" + nodeLabel + ":" + targetCellsLabel
        self.netParams.popParams[label] = {
            'cellType': 'art_VecStim',
            'numCells': number,
            'spkTimes': [0]
        }

    def createDevice(self, model, params):
        # TODO: For input device (differ by model): params.label contains name  of other node to connect to
        # For output device: params.label contains name of given node
        print(f"Netpyne:: will create internal device: {model} --- {params}")
        return NetpyneProxyDevice(netpyne_instance=self)

    def cellGids(self, population_label):
        return sim.net.pops[population_label].cellGids    

    def latestSpikes(self, timeWind):

        t = h.t # TODO: were and in other places: don't use h.t directly, but through netpyne engine

        spktvec = np.array(sim.simData['spkt'])
        spkgids = np.array(sim.simData['spkid'])
        # TODO: seems we don't need spktvec < t, because it's always True
        inds = np.nonzero((spktvec < t) * (spktvec > t - timeWind)) # filtered by time

        spktvec = spktvec[inds]
        spkgids = spkgids[inds]

        return spktvec, spkgids
    
    def cellGidsForPop(self, popLabel):
        return sim.net.pops[popLabel].cellGids

    def run(self, length):

        runForInterval(length)

        # TODO: to tun traditional runSimWithIntervalFunc in other thread (probably won't be used)
        # if self.started:
        #     print("Netpyne:: run next chunk")
        # else:
        #     print("Netpyne:: will start")

        # self.started = True

        # sim.prepareContinuousRun()

        # def runBgSimulation(interval_length, condition):
        #     tvbToNetpyneDtRatio = 0.1 # TODO: de-hardcode this ratio
        #     self.simConfig.duration = interval_length * 12345 # # Duration of the simulation, in ms
        #     self.simConfig.dt = interval_length * tvbToNetpyneDtRatio

        #     def intervalFun(time):
        #         condition.acquire()
        #         print(f"Hello! {time}")
        #         condition.notifyAll()
        #         condition.release()
            
        #     sim.runSimWithIntervalFunc(100, intervalFun)
        #     self.started = False


        # from threading import Thread, Condition
        # cond = Condition()
        # cond.acquire()
        # thread = Thread(target=runBgSimulation, name="netpyne_bg_simulation", args=(length, cond,))
        # thread.start()
        # cond.wait()
        # cond.release()
        
        # # sim.runSim()
        # # data = sim.gatherData()
        # print("Netpyne:: done")

from neuron import h
class NetpyneProxyDevice(object):

    netpyne_instance = None

    def __init__(self, netpyne_instance):
        self.netpyne_instance = netpyne_instance

    def convertFiringRate(self, values_dict, sourcePop, targetPop):

        rate = list(values_dict.values())[0]

        # print(f"Netpyne:: apply firing rate {rate}: {sourcePop} -> {targetPop}")

        stimulusPopLable = sourcePop + ":" + targetPop
        stimulusCellGids = sim.net.pops[stimulusPopLable].cellGids

         # TODO: make sure that rate should be divided by neurons number
        rate /= len(stimulusCellGids)

        spikesPerNeuron =  generateSpikesForPopulation(len(stimulusCellGids), rate, 0.1) # TODO: de-hardcode dt
        for index, spikes in spikesPerNeuron:
            spikes = spikes + h.t

            cell = sim.net.cells[stimulusCellGids[index]]
            vec = h.Vector()
            # print(f"Netpyne:: set {spikes} spikes at rate {rate} to {stimulusPopLable} (while t is {h.t})")
            cell.hPointp.play(vec.from_python(spikes))


    def neuronsConnecting(self, internalPopulation, externalNode=None):
        if externalNode is None:
            popLabel = internalPopulation
        else:
            popLabel = externalNode + ":" + internalPopulation
        return self.netpyne_instance.cellGids(popLabel)

    def numberOfSpikes(self, popLabel):

        # use only node (proper one), exclude artif cells' spikes
        cellGids = sim.net.pops[popLabel].cellGids

        t = h.t
        timewind = 0.1
        print(f"Netpyne:: will give data back. {popLabel} -- {t} -- {timewind}")

        spktvec = np.array(sim.simData['spkt'])
        spkgids = np.array(sim.simData['spkid'])

        timeFilter = (spktvec < t) * (spktvec > t - timewind)
        cellsFilter = np.isin(spkgids, cellGids)
        inds = np.nonzero(timeFilter * cellsFilter)

        spktvec = spktvec[inds]
        spkgids = spkgids[inds]

        num = len(spktvec)
        print(f"spikes nums >> {num}")
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

def runForInterval(interval):
    if round(h.t) < sim.cfg.duration:
        sim.pc.psolve(min(sim.cfg.duration, h.t+interval))
    else:
        sim.pc.barrier() # Wait for all hosts to get to this point
        sim.timing('stop', 'runTime')
        if sim.rank==0:
            print(('  Done; run time = %0.2f s; real-time ratio: %0.2f.' %
                (sim.timingData['runTime'], sim.cfg.duration/1000/sim.timingData['runTime'])))

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
