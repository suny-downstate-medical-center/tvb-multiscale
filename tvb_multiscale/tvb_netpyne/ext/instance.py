from random import randint
from ANNarchy.core.Simulate import simulate
import netpyne
import numpy as np
from tvb_multiscale.tvb_netpyne.ext.NodeCollection import NodeCollection

from netpyne import specs, sim
from netpyne.sim import *

# TODO: NetpyneInstance is stub
class NetpyneInstance(object):
    
    def __init__(self):
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
        simConfig.dt = 0.0025                # Internal integration timestep to use # TODO: should be decreased
        # simConfig.verbose = True           # Show detailed messages

        # simConfig.recordCells = ['uE']

        # simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
        simConfig.recordStep = 0.01          # Step size in ms to save data (eg. V traces, LFP, etc)
        simConfig.recordCellsSpikes = ['parahippocampal_L.E', 'parahippocampal_R.E'] # TODO: de-hardcode!
        # simConfig.filename = 'tut2'  # Set file output name
        simConfig.savePickle = False        # Save params, network and sim output to pickle file
        simConfig.saveJson = False

        self.simConfig = simConfig

        ## Population parameters
        
        # netParams.popParams['I'] = {'cellType': 'PYR', 'numCells': 20}

        # simConfig = specs.SimConfig()

        # sim.create(netParams = netParams, simConfig = simConfig)

    def createCells(self):
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

    def createExternalConnection(self, sourcePop, targetPop):
        import sys
        zeroRate = sys.float_info.min
        # nodes_connections are connections between given spiking node and other nodes
        # they are modelled as connections to artificial cells

        sourcePop = sourcePop + ":" + targetPop
        connLabel = sourcePop + '->' + targetPop
        # connParams = {
        #     'preConds': {'pop': sourcePop}, 
        #     'postConds': {'pop': targetPop},
        #     # 'probability': 1,
        #     'weight': 1.1,
        #     'synMech': 'exc',
        #     'delay': 0.0}

        self.netParams.connParams[connLabel] = {
            'preConds': {'pop': sourcePop},
            'postConds': {'pop': targetPop},
            'convergence': 1,
            'weight': 'uniform(0, 0.01)',
            'delay': 0,
            'synMech': 'exc'
        }
        # self.netParams.addConnParams(connLabel, connParams)

    def createInternalConnection(self, sourcePopulation, targetPopulation, synapticMechanism):
        if sourcePopulation.find(".E") > 0:
            synapticMechanism = 'exc'
        else:
            synapticMechanism = 'inh'
        label = sourcePopulation + "->" + targetPopulation
        # TODO: set proper values for weight, delay, etc.
        self.netParams.connParams[label] = {
            'preConds': {'pop': sourcePopulation},       # conditions of presyn cells
            'postConds': {'pop': targetPopulation},      # conditions of postsyn cells
            'probaility': 0.3,               # probability of connection
            'weight': 0.01,                 # synaptic weight
            'delay': 0.5,                     # transmission delay (ms)
            'synMech': synapticMechanism}               # synaptic mechanism
        pass

    def createNodeCollection(self, nodeLabel, popLabel, cellModel, number, params):
        print(f"Netpyne:: Creating population in node {nodeLabel} labeled '{popLabel}' of {number} neurons of type '{cellModel}'.")
        label = nodeLabel + "." + popLabel
        self.netParams.popParams[label] = {'cellType': cellModel, 'numCells': number}
        return NodeCollection(label, cellModel, number, params)

    from random import randint
    def createArtificialCells(self, nodeLabel, projectToNode, projectToPop, number, params=None):
        print(f"Netpyne:: Creating artif cells for node {nodeLabel} of {number} neurons.")
        targetCellsLabel = projectToNode + "." + projectToPop
        # TODO: this R_e should be de-hardcoded. Or worked around somehow else. Now it's given in test.py
        label = "R_e_" + nodeLabel + ":" + targetCellsLabel
        self.netParams.popParams[label] = {
            'cellType': 'art_VecStim',
            'numCells': number,
            'rate': 10,
            'start': randint(5, 45),
            'noise': 0.2
            # 'spkTimes': [.....]
        }

    def createDevice(self, model, params):
        # TODO: For input device (differ by model): params.label contains name  of other node to connect to
        # For output device: params.label contains name of given node
        print(f"Netpyne:: will create internal device: {model} --- {params}")
        return NetpyneProxyDevice(netpyne_instance=self)

    def cellGids(self, population_label):

        if population_label not in sim.net.pops:
            print(f"Netpyne network contains no population {population_label}!")
            return []
        gids = sim.net.pops[population_label].cellGids    
        if len(gids) == 0:
            print(f"Netpyne population {population_label} has no neurons!")
            return []
        return gids

    def latestSpikes(self, timeWind):

        t = h.t

        spktvec = np.array(sim.simData['spkt'])
        spkgids = np.array(sim.simData['spkid'])

        inds = np.nonzero((spktvec < t) * (spktvec > t - timeWind)) # filtered by time

        spktvec = spktvec[inds]
        spkgids = spkgids[inds]

        return spktvec, spkgids
    
    def cellGidsForPop(self, popLabel):
        return sim.net.pops[popLabel].cellGids

    def startSimulation(self, length):

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

class NetpyneProxyDevice(object):

    netpyne_instance = None

    def __init__(self, netpyne_instance):
        self.netpyne_instance = netpyne_instance

    def convertFiringRate(self, values_dict, node_label):

        rate = list(values_dict.values())[0]
        # print(f"Netpyne:: convert firing rate! {rate} for {node_label}")

        # TODO: uncomment once sanity-check of spikes done
        return

        # TODO: use proper label (for params?)
        popLabel = "E"
        interval = 1e3/rate

        def isStimulusFromNode(stim):
            postfix = stim["source"].split("stim-")[1] # TODO: de-hardcode "stim-"
            return postfix == node_label
        for gid in sim.net.pops[popLabel].cellGids:
            cell = sim.net.cells[gid]
            connectionsFromGivenNode = filter(isStimulusFromNode, cell.stims)
            for con in connectionsFromGivenNode:
                con['hObj'].interval = interval

        # for gid in sim.net.pops[popLabel].cellGids:
        #     cell = sim.net.cells[gid]
            # cell.stims[0]['hObj'].interval = interval

    # Rin_e_parahippocampal_L
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
from neuron import h
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
