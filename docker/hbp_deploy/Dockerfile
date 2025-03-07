#
# TVB & NEST simulations environment for OpenShift
#
# Prepare conda env, install NEST, then tvb-library from Pypi
# Finally tvb-multiscale from Github sources
#

FROM jupyter/minimal-notebook:2022-02-17
MAINTAINER lia.domide@codemart.ro

ENV LANG=C.UTF-8

USER root
RUN apt-get update; apt-get install -y build-essential cmake git vim octave wget  \
                            libltdl7-dev libreadline6-dev libncurses5-dev libgsl0-dev openmpi-bin libopenmpi-dev \
                            zip unzip; \
    apt-get clean

USER $NB_UID
RUN mkdir -m777 $HOME/packages

# Create a dedicated Python env for working in it
RUN conda create -y --name env_nest setuptools cython h5py python=3.9
RUN conda install -y --name env_nest pandas pytest pytest-cov pytest-benchmark sqlalchemy
RUN conda install -y --name env_nest numba matplotlib
RUN /opt/conda/envs/env_nest/bin/pip install --upgrade pip
RUN /opt/conda/envs/env_nest/bin/pip install xarray oct2py elephant scikit-learn scipy

WORKDIR $HOME/packages
RUN wget https://zenodo.org/record/3688773/files/tvb-data.zip?download=1 -O tvb_data.zip; \
    mkdir tvb_data; unzip tvb_data.zip -d tvb_data; rm tvb_data.zip; \
    cd tvb_data; \
    /opt/conda/envs/env_nest/bin/python setup.py develop

# Install NEST3, specific dev version (October 2020)
ENV NEST_SHA=c545255f7252308d6e73a851ddeece4f2dffc812
ENV NEST=nest-simulator
ENV NEST_SRC=$HOME/packages/$NEST

USER root
ARG NEST_BRANCH=master
RUN git clone --depth 1 --branch $NEST_BRANCH https://github.com/nest/nest-simulator.git
RUN cd $NEST; \
    git fetch origin && git checkout $NEST_SHA && cd ..; \
    source activate env_nest && cmake $NEST_SRC; \
    make; make install

USER $NB_UID
ARG TVB_BRANCH=master
RUN git clone --depth 1 --branch $TVB_BRANCH https://github.com/the-virtual-brain/tvb-root.git
ARG LAST_SHA_TVB=LATEST
RUN cd tvb-root; \
    git pull;\
    cd tvb_library; \
    /opt/conda/envs/env_nest/bin/python setup.py develop; \
    cd ../tvb_framework; \
    /opt/conda/envs/env_nest/bin/python setup.py develop; \
    cd ../tvb_contrib; \
    /opt/conda/envs/env_nest/bin/python setup.py develop;

# Install extra tvb-multiscale packages:

RUN git clone --depth 1 https://github.com/mariomulansky/PySpike.git; \
    cd PySpike; \
    /opt/conda/envs/env_nest/bin/python setup.py build_ext install

RUN /opt/conda/envs/env_nest/bin/pip install elephant dill

# TVB-NETPYNE:
RUN /opt/conda/envs/env_nest/bin/pip install netpyne neuron

# TVB-ANNARCHY:
RUN /opt/conda/envs/env_nest/bin/pip install sympy pyqtgraph lxml pandoc tensorboardX;
RUN /opt/conda/envs/env_nest/bin/pip install ANNarchy;

ARG MULTISCALE_BRANCH=master
RUN git clone --depth 1 --branch $MULTISCALE_BRANCH https://github.com/the-virtual-brain/tvb-multiscale.git
ARG LAST_SHA_MULTISCALE=LATEST
RUN cd tvb-multiscale; \
    git pull; \
    /opt/conda/envs/env_nest/bin/python setup.py develop --no-deps;

# Compile tvb-netpyne .mod files
RUN cd tvb-multiscale; \
    /opt/conda/envs/env_nest/bin/nrnivmodl tvb_multiscale/tvb_netpyne/netpyne/mod;

RUN mkdir -p -m777 /home/jovyan/.tvb-temp /home/jovyan/.local
RUN mkdir -p -m777 /home/jovyan/.local/share
RUN mkdir -p -m777 /home/jovyan/.local/share/jupyter
RUN mkdir -p -m777 /home/jovyan/.local/share/jupyter/runtime

WORKDIR $HOME

## Install local tvb-multiscale:
ENV MYMODULES_DIR=$HOME/packages/tvb-multiscale/tvb_multiscale/tvb_nest/nest/modules
ENV MYMODULES_BLD_DIR=$HOME/packages/nest_modules_builds
RUN cp -r ${MYMODULES_DIR} ${MYMODULES_BLD_DIR}

ARG WORK_DIR=/home/jovyan/packages/notebooks
ARG TVB_NEST_Examples=/home/jovyan/packages/notebooks/TVB-Multiscale-Examples

RUN mkdir -m777 $WORK_DIR
RUN mkdir -m777 $TVB_NEST_Examples
RUN cp -r /home/jovyan/packages/tvb-multiscale/docs/notebooks $TVB_NEST_Examples
RUN cp -r /home/jovyan/packages/tvb-multiscale/examples $TVB_NEST_Examples
RUN chmod -R 777 $WORK_DIR
RUN chmod -R 777 $TVB_NEST_Examples
RUN chmod -R 777 /home/jovyan/packages/tvb-multiscale/examples/tvb_nest
RUN chown -R jovyan: $TVB_NEST_Examples
RUN chown -R jovyan: /home/jovyan/packages/tvb-multiscale/examples/tvb_nest

USER root
RUN chmod -R 777 /home/jovyan/.jupyter
RUN chown -R jovyan: /home/jovyan/.jupyter
USER $NB_UID

# Extend the output data rate limit for all images to be loaded in the notebooks
#RUN echo "c.NotebookApp.iopub_data_rate_limit = 10000000" >> /etc/jupyter/jupyter_notebook_config.py
#RUN echo "c.NotebookApp.trust_xheaders = True" >> /etc/jupyter/jupyter_notebook_config.py

RUN /opt/conda/envs/env_nest/bin/python -m ipykernel install --user

# Necessary only because they are retrieved in tvb-multiscale code
ENV NEST_INSTALL_DIR=/opt/conda/envs/env_nest
ENV NEST_PYTHON_PREFIX=${NEST_INSTALL_DIR}/lib/python3.10/site-packages

WORKDIR $WORK_DIR

# CMD is defined in the base image: jupyter/minimal-notebook