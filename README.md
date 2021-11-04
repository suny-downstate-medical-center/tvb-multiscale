# TVB multiscale
This module currently provides a solution to interface between TVB and NetPyNE spiking networks for multiscale co-simulations.

# Project structure
At the top-level, we have the following folders:
## docker
Set of files used for building and distributing the module.

## docs
Here is were you can find some documentation about the module. In several forms: pdf or a jupyter notebook with documented steps. 

## examples
Set of scripts and jupyter notebooks that act as demos on how to use the API with different use-cases.

## tests
Unit and integration tests for the module.

## tvb_multiscale
This holds the whole codebase. In the below diagram, you can see a representation of the dependencies between the sub-folders:

                core
                /  \
               /    \
        tvb-netpyne  (.. other spiking simulators)


Description of sub-folders:

### core
Contains the base code that is considered generic/abstract enough to interface between a spiking network simulator and TVB (inside spiking_models and interfaces).

Here, we also keep I/O related code (read/write from/to H5 format and plots).

### tvb_netpyne
Code for interfacing with NetPyNE - depends on core and extends the classes defined there in order to specialize them for NetPyNE (inside netpyne_models and interfaces).

### tvb_elephant
Code that interfaces with Elephant and implements a wrapper around it that generates a co-simulator compatible stimulus.
