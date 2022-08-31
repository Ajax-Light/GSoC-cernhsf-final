dump_events: dump_events.cpp
	g++ -g -O0 -Wall -Wl,--copy-dt-needed-entries -std=c++17 -L /work/juggler/build -o dump_events dump_events.cpp \
	-lpodioRootIO \
	-leicd \
	-lJugRecoPlugins

	# g++ -g -O2 -Wall -std=c++17 -o dump_events dump_events.cpp -lpodio -lpodioRootIO -leicd $(root-config --glibs)
	
cc: CalClustering.cpp CalorimeterIslandCluster.cpp
	dpcpp -g -O0 -Wall -Wl,--copy-dt-needed-entries -std=c++17 -L /work/juggler/build -o cc CalClustering.cpp CalorimeterIslandCluster.cpp \
	-lpodioRootIO \
	-leicd \
	-lJugRecoPlugins \
	-lfmt

clean:
	rm -rf dump_events cc test_clusterop.root

dump_events.cpp:
CalClustering.cpp:
CalorimeterIslandCluster.cpp: