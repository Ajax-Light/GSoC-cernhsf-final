# SYCL Profiling and Testing notes

This repo includes Benchmarking/Testing code for the SYCL-ported [IslandClusterAlgorithm](https://eicweb.phy.anl.gov/EIC/juggler/-/blob/master/JugReco/src/components/CalorimeterIslandCluster.cpp)

## Issues

At the time of writing, the following issues prevent proper profiling of the SYCL code:

* Intel VTune does not work with Python (used by `gaudirun.py`) inside a container
* There are plans to move from **Gaudi** to **JANA2** for algorithm execution

Considering the above, we have decided to first rewrite the Algorithm as functionals, then proceed to profile it.
Based on the profile, the algorithm is rewritten in SYCL and changes are pushed to the upstream Juggler repo.

## Directory Structure

1. Source code (src):

    * `CalClustering.cpp` contains the Main function
    * `CalorimeterIslandCluster.cpp` is the Island Cluster Algorithm which has been separated into functionals
    * `AlgoHeaders.h` contains Class definitions for the algorithms, Gaudi's Units and Properties struct

    a. Misc Files:

        * `Makefile` has a target `cc` which invokes the clustering algorithm. Build using `make cc`
        * `notes.txt` contains notes regarding the build process and issues encountered during setting up the build system. Refer this when program execution fails.

2. Reports:

    * [Wall Time comparision with and without SYCL](./reports/WallTime-compare.png)
    * [Screenshots of VTune Profiler Results](./reports/perf_cmpr.pdf)
