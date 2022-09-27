export LIBRARY_PATH=/work/Gaudi/build:/work/juggler/build:$$LIBRARY_PATH
test:
	g++ -g -std=c++17 -L /work/Gaudi/build GaudiMain.cpp -o test \
	-lGaudiKernel \
	-lRootCnvLib \
	-lJugRecoPlugins \
	-lJugBase

clean:
	rm -rf test

