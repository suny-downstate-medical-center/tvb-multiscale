#!/bin/sh

echo "Pulling TVB-Multiscale docker image"
/apps/daint/system/opt/sarus/1.1.0/bin/sarus pull thevirtualbrain/tvb-multiscale:2.0.1

start=$SECONDS

echo "Start docker container"
srun -C mc /apps/daint/system/opt/sarus/1.1.0/bin/sarus --debug run --mpi --mount=type=bind,source=$PWD,destination=$PWD thevirtualbrain/tvb-multiscale:2.0.1 /bin/bash -c "cd $PWD && /home/docker/env/neurosci/bin/python $1"

duration=$(( SECONDS - start ))

echo "TVB-NEST test completed in $duration seconds"
