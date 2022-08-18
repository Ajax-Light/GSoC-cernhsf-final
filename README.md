# SYCL Profiling and Testing notes

This repo includes Benchmarking/Testing code for the SYCL-ported [IslandClusterAlgorithm](https://eicweb.phy.anl.gov/EIC/juggler/-/blob/master/JugReco/src/components/CalorimeterIslandCluster.cpp)

## Issues

At the time of writing, the following issues prevent proper profiling of the SYCL code:

* Intel VTune does not work with Python (used by `gaudirun.py`) inside a container
* There are plans to move from Gaudi to JANA2 for algorithm execution

Considering the above, we have decided to first rewrite the Algorithm as functionals, then proceed to profile it.
Based on the profile, the algorithm is rewritten in SYCL and changes are pushed to the upstream Juggler repo.
