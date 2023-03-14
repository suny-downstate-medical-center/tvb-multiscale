import numpy as np

from netpyne import specs, sim
from netpyne.sim import *

# performance with time
import time
import logging

logger = logging.getLogger('benchmark')

class NetpyneModule(object):

    spikeGenerators = []
    
    def __init__(self):
        self.spikeGeneratorPops = []
        self.autoCreatedPops = []

        # Make sure that all required mod-files are compiled (is there a better way to check?)
        try:
            h.DynamicNetStim()
        except:
            print("NetPyNE couldn't find necessary MOD-files. Trying to compile..")
            import sys
            import os
            python_path = sys.executable.split("python")[0]
            tvb_multiscale_path = os.path.abspath(__file__).split("tvb_multiscale")[0]
            os.system('%snrnivmodl %s/tvb_multiscale/tvb_netpyne/netpyne/mod' % (python_path, tvb_multiscale_path))
            h.nrn_load_dll('./x86_64/libnrnmech.so')

    def importModel(self, netParams, simConfig, dt, config):

        simConfig.dt = dt
        simConfig.duration = config.simulation_length

        simConfig.simLabel = 'spiking'
        simConfig.saveFolder = config.out._out_base # TODO: better use some public method

        self.netParams = netParams
        self.simConfig = simConfig

        # using DynamicNetStim model for artificial cells serving as stimuli
        self.netParams.cellParams['art_NetStim'] = {'cellModel': 'DynamicNetStim'}

    @property
    def dt(self):
        return self.simConfig.dt

    @property
    def minDelay(self):
        return self.dt

    @property
    def time(self):
        return h.t
    
    def instantiateNetwork(self):

        sim.initialize(self.netParams, None) # simConfig to be provided later
        sim.net.createPops()
        sim.net.createCells()

        if len(self.autoCreatedPops):
            # choose N random cells from each population to plot traces for
            n = 1
            rnd = np.random.RandomState(0)
            def includeFor(pop):
                popSize = len(sim.net.pops[pop].cellGids)
                chosen = (pop, rnd.choice(popSize, size=min(n, popSize), replace=False).tolist())
                return chosen
            include = list(map(includeFor, self.autoCreatedPops))
            self.simConfig.analysis['plotTraces'] = {'include': include, 'saveFig': True}
            self.simConfig.analysis['plotRaster'] = {'saveFig': True, 'include': self.autoCreatedPops, 'popRates': 'minimal'}

            if self.simConfig.recordCellsSpikes == -1:
                allPopsButSpikeGenerators = [pop for pop in self.netParams.popParams.keys() if pop not in self.spikeGeneratorPops]
                self.simConfig.recordCellsSpikes = allPopsButSpikeGenerators

        sim.setSimCfg(self.simConfig)
        sim.setNetParams(self.netParams)
        sim.net.params.synMechParams.preprocessStringFunctions()
        sim.net.params.cellParams.preprocessStringFunctions()

        sim.net.connectCells()
        sim.net.addStims()
        sim.net.addRxD()
        sim.setupRecording()

        sim.run.prepareSimWithIntervalFunc()

        # interval run is used internally to support communication with TVB. However, users may also need to have feedbacks
        # at some interval, distinct from those used internally. User-defined interval and intervalFunc are read here, and the logic is handled in `run()`
        if hasattr(self.simConfig, 'interval'):
            self.interval = self.simConfig.interval
            self.intervalFunc = self.simConfig.intervalFunc
            self.nextIntervalFuncCall = self.interval
        else:
            self.nextIntervalFuncCall = None

    def connectStimuli(self, sourcePop, targetPop, weight, delay, receptorType):
        # TODO: randomize weight and delay, if values do not already contain sting func
        # (e.g. use random_normal_weight() and random_uniform_delay() from netpyne_templates)
        sourceCells = self.netParams.popParams[sourcePop]['numCells']
        targetCells = self.netParams.popParams[targetPop]['numCells']

        # connect cells roughly one-to-one ('lamda' for E -> I connections is already taken into account, as it baked into source population size)
        if sourceCells <= targetCells:
            rule = 'divergence'
        else:
            rule = 'convergence'

        connLabel = sourcePop + '->' + targetPop
        self.netParams.connParams[connLabel] = {
            'preConds': {'pop': sourcePop},
            'postConds': {'pop': targetPop},
            rule: 1.0,
            'weight': weight,
            'delay': delay,
            'synMech': receptorType
        }

    def interconnectSpikingPopulations(self, sourcePopulation, targetPopulation, synapticMechanism, weight, delay, probabilityOfConn):

        label = sourcePopulation + "->" + targetPopulation
        self.netParams.connParams[label] = {
            'preConds': {'pop': sourcePopulation},
            'postConds': {'pop': targetPopulation},
            'probability': probabilityOfConn,
            'weight': weight,
            'delay': delay,
            'synMech': synapticMechanism }

    def registerPopulation(self, label, cellModel, size):
        self.autoCreatedPops.append(label)
        self.netParams.popParams[label] = {'cellType': cellModel, 'numCells': size}

    def createArtificialCells(self, label, number, params=None):
        print(f"Netpyne:: Creating artif cells for node '{label}' of {number} neurons")
        self.spikeGeneratorPops.append(label)
        self.netParams.popParams[label] = {
            'cellType': 'art_NetStim',
            'numCells': number,
        }

    def getSpikes(self, generatedBy=None, startingFrom=None):
        tic = time.time()
        spktimes = np.array(sim.simData['spkt'])
        spkgids = np.array(sim.simData['spkid'])

        if startingFrom is not None:
            inds = np.nonzero(spktimes > startingFrom) # filtered by time # (self.time - timeWind)

            spktimes = spktimes[inds]
            spkgids = spkgids[inds]

        if generatedBy is not None:
            inds = np.isin(spkgids, generatedBy)

            spktimes = spktimes[inds]
            spkgids = spkgids[inds]
        logger.log(level=200, msg="getSpikes, {}".format(time.time() - tic))
        return spktimes, spkgids
    
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
        tic = time.time()

        self.stimulate(length)

        # handling (potentially) two distinct intervals (see comment in `createAndPrepareNetwork()`)
        tvbIterationEnd = self.time + length
        def _(simTime): pass
        if self.nextIntervalFuncCall:
            while (self.nextIntervalFuncCall < tvbIterationEnd):
                if self.time < sim.cfg.duration:
                    sim.run.runForInterval(self.nextIntervalFuncCall - self.time, _)
                    self.intervalFunc(self.time)
                    self.nextIntervalFuncCall = self.time + self.interval
                else:
                    break
        if tvbIterationEnd > self.time:
            if self.time < sim.cfg.duration:
                sim.run.runForInterval(tvbIterationEnd - self.time, _)

                if self.nextIntervalFuncCall:
                    # add correction to avoid accumulation of arithmetic error due to that h.t advances slightly more than the requested interval
                    correction = self.time - tvbIterationEnd
                    self.nextIntervalFuncCall += correction
        logger.log(level=200, msg="run, {}".format(time.time() - tic))

    def stimulate(self, length):
        tic = time.time()
        allNeuronsSpikes = {}
        allNeurons = []
        for device in self.spikeGenerators:
            allNeurons.extend(device.own_neurons)
            allNeuronsSpikes.update(device.spikesPerNeuron)

            device.spikesPerNeuron = {} # clear to prepare for next interval run

        intervalEnd = h.t + length
        for gid in allNeurons:
            # currently used .mod implementation allows no more then 3 spikes during the interval.
            # if for given cell they are less than 3, use -1 for the rest. If they are more, the rest will be lost. But for the reasonable spiking rates, this latter case is highly unlikely. 
            spikes = allNeuronsSpikes.get(gid, [])
            spks = [-1] * 3
            for i, spike in enumerate(spikes[:3]):
                spks[i] = spike
            sim.net.cells[gid].hPointp.set_next_spikes(intervalEnd, spks[0], spks[1], spks[2])
        logger.log(level=200, msg="stimulate, {}".format(time.time() - tic))

    def finalize(self):
        if self.time < sim.cfg.duration:
            stopTime = None
        else:
            stopTime = sim.cfg.duration
        sim.run.postRun(stopTime)
        sim.gatherData()
        sim.analyze()
