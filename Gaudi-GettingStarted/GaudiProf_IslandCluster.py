import os
import sys

# Gaudi Imports
from Gaudi.Configuration import *
from Configurables import ApplicationMgr, EICDataSvc, PodioInput, PodioOutput, GeoSvc
from GaudiKernel.SystemOfUnits import MeV, GeV, mm, cm, mrad

# Parse environment variables (Detector and its compact description)
detector_name = str(os.environ.get("JUGGLER_DETECTOR_CONFIG", "athena"))
detector_version = str(os.environ.get("JUGGLER_DETECTOR_VERSION", "master"))

detector_path = str(os.environ.get("DETECTOR_PATH", "."))
compact_path = os.path.join(detector_path, detector_name)

# Gaudirun.py doesn't work with cmdline args :(

"""
if len(sys.argv) < 3:
        print("INFO: Defaulting to 1k run !!")
        profile_inp_size = "1k"
else:
        #profile_inp_size = str(sys.argv[-1])
        profile_inp_size = "10k"
"""
profile_inp_size="1k"

# Input / Output
## Input from Digi/Reco output
input_sims = ["rec_profile_DigiReco_{}.root".format(profile_inp_size)]
output_rec = "rec_profile_Clusters{}.root".format(profile_inp_size)
n_events = int(profile_inp_size[:profile_inp_size.find('k')]) * 1000
print(sys.argv,input_sims, output_rec, n_events)

# Geometry service
geo_service = GeoSvc("GeoSvc", detectors=["{}.xml".format(compact_path)], OutputLevel=INFO)

# Data service
podioevent = EICDataSvc("EventDataSvc", inputs=input_sims)

# Juggler Components
from Configurables import Jug__Digi__CalorimeterHitDigi as CalHitDigi
from Configurables import Jug__Reco__CalorimeterHitReco as CalHitReco
from Configurables import Jug__Reco__CalorimeterIslandCluster as IslandCluster

# branches needed from simulation root file
sim_coll = [
    "MCParticles",
    "EcalEndcapNHits",
    "EcalEndcapNHitsContributions",
    "EcalEndcapNHitsReco",
    "EcalEndcapPHits",
    "EcalEndcapPHitsContributions",
    "EcalBarrelHits",
    "EcalBarrelHitsContributions",
    "HcalBarrelHits",
    "HcalBarrelHitsContributions",
    "HcalEndcapPHits",
    "HcalEndcapPHitsContributions",
    "HcalEndcapNHits",
    "HcalEndcapNHitsContributions",
]


# List of algorithms to process
algs = []

# ====== Start Algorithm Listing =======

# Input
inp = PodioInput("PodioReader", collections=sim_coll)
algs.append(inp)

## Crystal Endcap Ecal
# 3. Cluster Hits
ce_ecal_cl = IslandCluster("ce_ecal_cl",
        inputHitCollection="EcalEndcapNHitsReco",
        outputProtoClusterCollection="EcalEndcapNProtoClusters",
        splitCluster=False,
        minClusterHitEdep=1.0*MeV,  # discard low energy hits
        minClusterCenterEdep=30*MeV,
        sectorDist=5.0*cm,
        dimScaledLocalDistXY=[1.8, 1.8]) # dimension scaled dist is good for hybrid sectors with different module size
algs.append(ce_ecal_cl)

# Output
out = PodioOutput("out", filename=output_rec)
out.outputCommands = ['drop *',
        'keep MCParticles',
        'keep *Digi',
        'keep *Reco*',
        'keep *Cluster*',
        'keep *Layers']
algs.append(out)

ApplicationMgr(
    TopAlg = algs,
    EvtSel = 'NONE',
    EvtMax = n_events,
    ExtSvc = [podioevent],
    OutputLevel=DEBUG
    )
