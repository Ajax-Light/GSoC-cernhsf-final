// podio specific includes
#include "podio/ROOTReader.h"
#include "podio/ROOTWriter.h"
#include "podio/EventStore.h"
#include "podio/IReader.h"
#include "podio/UserDataCollection.h"
#include "podio/podioVersion.h"

// DataModel (eicd) includes
#include "eicd/ClusterCollection.h"
#include "eicd/ProtoClusterCollection.h"
#include "eicd/ReconstructedParticleCollection.h"
#include "eicd/MCRecoClusterParticleAssociationCollection.h"

// STL
#include <cassert>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>

// Custom Headers
#include "AlgoHeaders.h"

void processEvent(podio::EventStore& store, int eventNum, struct props& p) {
    auto& recoPC = store.get<eicd::CalorimeterHitCollection>("EcalEndcapNHitsReco");
    
    // Call Calo Algos
    Jug::Reco::CalorimeterIslandCluster cis("CalorimeterIslandCluster", p, recoPC);
    cis.initialize();

    if(recoPC.isValid()){
        std::cout << "Event: " << eventNum << " ,EcalEndcapNHitsReco , size: " << recoPC.size() << "\n";
        cis.execute();
        /* if(proto.isValid()){
            for(auto i : proto){
                for(auto it = i.hits_begin(); it != i.hits_end(); it++){
                    std::cout << *it << "\n";
                }
            }
        }else {
            std::cout << "Size 0, skipping\n";
        } */
    } else {
        throw std::runtime_error("Collection 'EcalEndcapNHitsReco' should be present");
    }
}

void initProps(struct props& p){
    p.is_debug = true;
    p.is_verbose = true;

    p.m_splitCluster = true;
    p.m_minClusterHitEdep = 0.0;
    p.m_minClusterCenterEdep = 50.0 * my_units::MeV;

    p.m_sectorDist = 5.0 * my_units::cm;
    p.u_localDistXY = {};
    p.u_localDistXZ = {};
    p.u_localDistYZ = {};
    p.u_globalDistRPhi = {};
    p.u_globalDistEtaPhi = {};
    p.u_dimScaledLocalDistXY = {1.8, 1.8};
}

int main(int argc, char** argv) {
    std::string inp;
    if(argc == 1){
        std::cout << "Please specify input ROOT file \n";
        return 1;
    }else{
        inp = argv[1];
    }
    
    struct props p;
    initProps(p);

    auto reader = podio::ROOTReader();
    reader.openFile(inp);

    auto store = podio::EventStore();
    store.setReader(&reader);

    /* auto writer = podio::ROOTWriter("test_clusterop.root", &store);
    auto& proto = store.create<eicd::ProtoClusterCollection>("EcalEndcapNProtoClusters");
    writer.registerForWrite("EcalEndcapNProtoClusters"); */

    const auto nEvents = reader.getEntries();

    for (unsigned i = 0; i < nEvents; ++i) {

        if (i % 1000 == 0) {
        std::cout << "reading event " << i << std::endl;
        }

        // Process Event
        processEvent(store, i, p);

        //writer.writeEvent();
    
        store.clear();
        reader.endOfEvent();
    }

    reader.closeFile();
    return 0;
}
