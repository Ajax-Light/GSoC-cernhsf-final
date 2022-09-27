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
if len(sys.argv) < 3:
        print("INFO: Defaulting to 1k run !!")
        profile_inp_size = "1k"
else:
        #profile_inp_size = str(sys.argv[-1])
        profile_inp_size = "10k"

# Input / Output
input_sims = ["sim_profile_{}.edm4hep.root".format(profile_inp_size)]
output_rec = "rec_profile_DigiReco_{}.root".format(profile_inp_size)
n_events = int(profile_inp_size[:profile_inp_size.find('k')]) * 1000
#print(input_sims, output_rec, n_events)

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

# Crystal Endcap Ecal
ce_ecal_daq = dict(
        dynamicRangeADC=5.*GeV,
        capacityADC=32768,
        pedestalMean=400,
        pedestalSigma=3)

# 1. Digitize Hits
ce_ecal_digi = CalHitDigi("ce_ecal_digi",
        inputHitCollection="EcalEndcapNHits",
        outputHitCollection="EcalEndcapNHitsDigi",
        energyResolutions=[0., 0.02, 0.],
        **ce_ecal_daq)
algs.append(ce_ecal_digi)

# 2. Reconstruct Hits
ce_ecal_reco = CalHitReco("ce_ecal_reco",
        inputHitCollection=ce_ecal_digi.outputHitCollection,
        outputHitCollection="EcalEndcapNHitsReco",
        thresholdFactor=4,          # 4 sigma cut on pedestal sigma
        readoutClass="EcalEndcapNHits",
        sectorField="sector",
        samplingFraction=0.998,      # this accounts for a small fraction of leakage
        **ce_ecal_daq)
algs.append(ce_ecal_reco)


# Output
out = PodioOutput("out", filename=output_rec)
out.outputCommands = ['keep *']
algs.append(out)

ApplicationMgr(
    TopAlg = algs,
    EvtSel = 'NONE',
    EvtMax = n_events,
    ExtSvc = [podioevent],
    OutputLevel=WARNING
    )
