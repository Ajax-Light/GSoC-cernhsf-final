// podio specific includes
#include "podio/ROOTReader.h"
#include "podio/EventStore.h"
#include "podio/IReader.h"
#include "podio/UserDataCollection.h"
#include "podio/podioVersion.h"

// DataModel (eicd) includes
#include "eicd/ClusterCollection.h"
#include "eicd/ProtoClusterCollection.h"
#include "eicd/ReconstructedParticleCollection.h"
#include "eicd/CalorimeterHitCollection.h"

// STL
#include <cassert>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>

void processEvent(podio::EventStore& store, int eventNum, podio::version::Version fileVersion) {
  auto& recoPC = store.get<eicd::CalorimeterHitCollection>("EcalEndcapNHitsReco");

  std::cout << recoPC[1].getCellID() << std::endl; // Why does << crash program?
  return;

  if(recoPC.isValid()){
    std::cout << "Event: " << eventNum << " ,EcalEndcapNHitsReco , size: " << recoPC.size() << "\n";
    for(size_t i = 0; i < recoPC.size(); i++){
      std::cout << "Particle: " << i <<" in Collection: " << eventNum << " Energy: " << recoPC[i].getEnergy() << "\n";
    }
  } else {
    throw std::runtime_error("Collection 'EcalEndcapNHitsReco' should be present");
  }

}

int main(int argc, char** argv) {
  std::string inp;
  if(argc == 1){
	  std::cout << "Please specify input ROOT file \n";
	  return 1;
  }else{
	  inp = argv[1];
  }
  
  auto reader = podio::ROOTReader();
  reader.openFile(inp);
  /* Skip Test for now
  if (reader.currentFileVersion() != podio::version::build_version) {
    std::cout << "Current File Version: " << reader.currentFileVersion() << std::endl;
    std::cout << "Podio Build Version: " << podio::version::build_version << std::endl;
    return 1;
  }
  */

  auto store = podio::EventStore();
  store.setReader(&reader);

  const auto nEvents = reader.getEntries();

  for (unsigned i = 0; i < nEvents; ++i) {

    if (i % 1000 == 0) {
      std::cout << "reading event " << i << std::endl;
    }

    // Process Event
    processEvent(store, i, reader.currentFileVersion());

    store.clear();
    reader.endOfEvent();
  }

  reader.closeFile();
  return 0;
}
