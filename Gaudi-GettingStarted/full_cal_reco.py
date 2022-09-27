'''
    An example option file to digitize/reconstruct/clustering calorimeter hits
'''
from Gaudi.Configuration import *
import json
import os
import ROOT

from Configurables import ApplicationMgr, EICDataSvc, PodioInput, PodioOutput, GeoSvc
from GaudiKernel.SystemOfUnits import MeV, GeV, mm, cm, mrad

detector_name = str(os.environ.get("JUGGLER_DETECTOR_CONFIG", "athena"))
detector_version = str(os.environ.get("JUGGLER_DETECTOR_VERSION", "master"))

detector_path = str(os.environ.get("DETECTOR_PATH", "."))
compact_path = os.path.join(detector_path, detector_name)

# input arguments from calibration file
with open(f'{detector_path}/calibrations/emcal_barrel_calibration.json') as f:
    calib_data = json.load(f)['electron']

print(calib_data)

cb_ecal_sf = float(calib_data['sampling_fraction_img'])
scifi_barrel_sf = float(calib_data['sampling_fraction_scfi'])

# get sampling fractions from system environment variable, 1.0 by default
ci_ecal_sf = float(os.environ.get("CI_ECAL_SAMP_FRAC", 0.253))
cb_hcal_sf = float(os.environ.get("CB_HCAL_SAMP_FRAC", 0.038))
ci_hcal_sf = float(os.environ.get("CI_HCAL_SAMP_FRAC", 0.025))
ce_hcal_sf = float(os.environ.get("CE_HCAL_SAMP_FRAC", 0.025))

# Gaudirun.py doesn't work with cmdline args :(
if len(sys.argv) < 3:
        print("INFO: Defaulting to 1k run !!")
        profile_inp_size = "1k"
else:
        #profile_inp_size = str(sys.argv[-1])
        profile_inp_size = "10k"

# Input / Output
input_sims = ["sim_profile_{}.edm4hep.root".format(profile_inp_size)]
output_rec = "rec_test_{}.root".format(profile_inp_size)
n_events = int(profile_inp_size[:profile_inp_size.find('k')]) * 1000
#print(input_sims, output_rec, n_events)

# geometry service
geo_service = GeoSvc("GeoSvc", detectors=["{}.xml".format(compact_path)], OutputLevel=INFO)
# data service
podioevent = EICDataSvc("EventDataSvc", inputs=input_sims)


# juggler components
from Configurables import Jug__Digi__CalorimeterHitDigi as CalHitDigi
from Configurables import Jug__Reco__CalorimeterHitReco as CalHitReco
from Configurables import Jug__Reco__CalorimeterHitsMerger as CalHitsMerger
from Configurables import Jug__Reco__CalorimeterIslandCluster as IslandCluster

from Configurables import Jug__Reco__ImagingPixelReco as ImCalPixelReco
from Configurables import Jug__Reco__ImagingTopoCluster as ImagingCluster

from Configurables import Jug__Reco__ClusterRecoCoG as RecoCoG
from Configurables import Jug__Reco__ImagingClusterReco as ImagingClusterReco

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

if 'athena' in detector_name:
    sim_coll += [
        "EcalBarrelScFiHits",
        "EcalBarrelScFiHitsContributions",
    ]

# list of algorithms
algs = []

# input
podin = PodioInput("PodioReader", collections=sim_coll)
algs.append(podin)

# Crystal Endcap Ecal
ce_ecal_daq = dict(
        dynamicRangeADC=5.*GeV,
        capacityADC=32768,
        pedestalMean=400,
        pedestalSigma=3)

ce_ecal_digi = CalHitDigi("ce_ecal_digi",
        inputHitCollection="EcalEndcapNHits",
        outputHitCollection="EcalEndcapNHitsDigi",
        energyResolutions=[0., 0.02, 0.],
        **ce_ecal_daq)
algs.append(ce_ecal_digi)

ce_ecal_reco = CalHitReco("ce_ecal_reco",
        inputHitCollection=ce_ecal_digi.outputHitCollection,
        outputHitCollection="EcalEndcapNHitsReco",
        thresholdFactor=4,          # 4 sigma cut on pedestal sigma
        readoutClass="EcalEndcapNHits",
        sectorField="sector",
        samplingFraction=0.998,      # this accounts for a small fraction of leakage
        **ce_ecal_daq)
algs.append(ce_ecal_reco)

ce_ecal_cl = IslandCluster("ce_ecal_cl",
        # OutputLevel=DEBUG,
        inputHitCollection=ce_ecal_reco.outputHitCollection,
        outputProtoClusterCollection="EcalEndcapNProtoClusters",
        splitCluster=False,
        minClusterHitEdep=1.0*MeV,  # discard low energy hits
        minClusterCenterEdep=30*MeV,
        sectorDist=5.0*cm,
        dimScaledLocalDistXY=[1.8, 1.8])          # dimension scaled dist is good for hybrid sectors with different module size
algs.append(ce_ecal_cl)

ce_ecal_clreco = RecoCoG("ce_ecal_clreco",
        inputProtoClusterCollection=ce_ecal_cl.outputProtoClusterCollection,
        outputClusterCollection="EcalEndcapNClusters",
        logWeightBase=4.6)
algs.append(ce_ecal_clreco)

# Endcap Sampling Ecal
ci_ecal_daq = dict(
        dynamicRangeADC=50.*MeV,
        capacityADC=32768,
        pedestalMean=400,
        pedestalSigma=10)

ci_ecal_digi = CalHitDigi("ci_ecal_digi",
        inputHitCollection="EcalEndcapPHits",
        outputHitCollection="EcalEndcapPHitsDigi",
        **ci_ecal_daq)
algs.append(ci_ecal_digi)

ci_ecal_reco = CalHitReco("ci_ecal_reco",
        inputHitCollection=ci_ecal_digi.outputHitCollection,
        outputHitCollection="EcalEndcapPHitsReco",
        thresholdFactor=5.0,
        samplingFraction=ci_ecal_sf,
        **ci_ecal_daq)
algs.append(ci_ecal_reco)

# merge hits in different layer (projection to local x-y plane)
ci_ecal_merger = CalHitsMerger("ci_ecal_merger",
        # OutputLevel=DEBUG,
        inputHitCollection=ci_ecal_reco.outputHitCollection,
        outputHitCollection="EcalEndcapPHitsRecoXY",
        # fields=["layer", "slice"],
        # fieldRefNumbers=[1, 0],
        fields=["fiber_x", "fiber_y"],
        fieldRefNumbers=[1, 1],
        readoutClass="EcalEndcapPHits")
algs.append(ci_ecal_merger)

ci_ecal_cl = IslandCluster("ci_ecal_cl",
        # OutputLevel=DEBUG,
        inputHitCollection=ci_ecal_merger.inputHitCollection,
        outputProtoClusterCollection="EcalEndcapPProtoClusters",
        splitCluster=False,
        minClusterCenterEdep=10.*MeV,
        localDistXY=[10*mm, 10*mm])
algs.append(ci_ecal_cl)

ci_ecal_clreco = RecoCoG("ci_ecal_clreco",
        inputProtoClusterCollection=ci_ecal_cl.outputProtoClusterCollection,
        outputClusterCollection="EcalEndcapPClusters",
        logWeightBase=6.2)
algs.append(ci_ecal_clreco)

# Central Barrel Ecal
if 'athena' in detector_name:
    # Imaging calorimeter
    cb_ecal_daq = dict(
        dynamicRangeADC=3*MeV,
        capacityADC=8192,
        pedestalMean=400,
        pedestalSigma=20)   # about 6 keV

    cb_ecal_digi = CalHitDigi("cb_ecal_digi",
        inputHitCollection="EcalBarrelHits",
        outputHitCollection="EcalBarrelImagingHitsDigi",
        energyResolutions=[0., 0.02, 0.],   # 2% flat resolution
        **cb_ecal_daq)
    algs.append(cb_ecal_digi)

    cb_ecal_reco = ImCalPixelReco("cb_ecal_reco",
        inputHitCollection=cb_ecal_digi.outputHitCollection,
        outputHitCollection="EcalBarrelImagingHitsReco",
        thresholdFactor=3,  # about 20 keV
        readoutClass="EcalBarrelHits",  # readout class
        layerField="layer",             # field to get layer id
        sectorField="module",           # field to get sector id
        samplingFraction=cb_ecal_sf,
        **cb_ecal_daq)
    algs.append(cb_ecal_reco)

    cb_ecal_cl = ImagingCluster("cb_ecal_cl",
        inputHitCollection=cb_ecal_reco.outputHitCollection,
        outputProtoClusterCollection="EcalBarrelImagingProtoClusters",
        localDistXY=[2.*mm, 2*mm],              # same layer
        layerDistEtaPhi=[10*mrad, 10*mrad],     # adjacent layer
        neighbourLayersRange=2,                 # id diff for adjacent layer
        sectorDist=3.*cm)                       # different sector
    algs.append(cb_ecal_cl)

    cb_ecal_clreco = ImagingClusterReco("cb_ecal_clreco",
        inputProtoClusters=cb_ecal_cl.outputProtoClusterCollection,
        mcHits="EcalBarrelHits",
        outputClusters="EcalBarrelImagingClusters",
        outputLayers="EcalBarrelImagingLayers")
    algs.append(cb_ecal_clreco)
else:
    # SciGlass calorimeter
    cb_ecal_daq = dict(
        dynamicRangeADC=5.*GeV,
        capacityADC=32768,
        pedestalMean=400,
        pedestalSigma=3)

    cb_ecal_digi = CalHitDigi("cb_ecal_digi",
        inputHitCollection="EcalBarrelHits",
        outputHitCollection="EcalBarrelHitsDigi",
        energyResolutions=[0., 0.02, 0.],   # 2% flat resolution
        **cb_ecal_daq)
    algs.append(cb_ecal_digi)

    cb_ecal_reco = CalHitReco("cb_ecal_reco",
        inputHitCollection=cb_ecal_digi.outputHitCollection,
        outputHitCollection="EcalBarrelHitsReco",
        thresholdFactor=3,  # about 20 keV
        readoutClass="EcalBarrelHits",  # readout class
        sectorField="sector",           # field to get sector id
        samplingFraction=0.998,         # this accounts for a small fraction of leakage
        **cb_ecal_daq)
    algs.append(cb_ecal_reco)

    cb_ecal_cl = IslandCluster("cb_ecal_cl",
        inputHitCollection=cb_ecal_reco.outputHitCollection,
        outputProtoClusterCollection="EcalBarrelProtoClusters",
        splitCluster=False,
        minClusterHitEdep=1.0*MeV,  # discard low energy hits
        minClusterCenterEdep=30*MeV,
        sectorDist=5.0*cm)
    algs.append(cb_ecal_cl)

    cb_ecal_clreco = ImagingClusterReco("cb_ecal_clreco",
        inputProtoClusters=cb_ecal_cl.outputProtoClusterCollection,
        mcHits="EcalBarrelHits",
        outputClusters="EcalBarrelClusters",
        outputLayers="EcalBarrelLayers")
    algs.append(cb_ecal_clreco)

# Central Barrel Ecal SciFi
if 'athena' in detector_name:
    scfi_barrel_daq = dict(
        dynamicRangeADC=50.*MeV,
        capacityADC=32768,
        pedestalMean=400,
        pedestalSigma=10)

    scfi_barrel_digi = CalHitDigi("scfi_barrel_digi",
        inputHitCollection="EcalBarrelScFiHits",
        outputHitCollection="EcalBarrelScFiHitsDigi",
        **scfi_barrel_daq)
    algs.append(scfi_barrel_digi)

    scfi_barrel_reco = CalHitReco("scfi_barrel_reco",
        inputHitCollection=scfi_barrel_digi.outputHitCollection,
        outputHitCollection="EcalBarrelScFiHitsReco",
        thresholdFactor=5.0,
        readoutClass="EcalBarrelScFiHits",
        layerField="layer",
        sectorField="module",
        localDetFields=["system", "module"], # use local coordinates in each module (stave)
        samplingFraction=scifi_barrel_sf,
        **scfi_barrel_daq)
    algs.append(scfi_barrel_reco)

    # merge hits in different layer (projection to local x-y plane)
    scfi_barrel_merger = CalHitsMerger("scfi_barrel_merger",
        # OutputLevel=DEBUG,
        inputHitCollection=scfi_barrel_reco.outputHitCollection,
        outputHitCollection="EcalBarrelScFiGridReco",
        fields=["fiber"],
        fieldRefNumbers=[1],
        readoutClass="EcalBarrelScFiHits")
    algs.append(scfi_barrel_merger)

    scfi_barrel_cl = IslandCluster("scfi_barrel_cl",
        # OutputLevel=DEBUG,
        inputHitCollection=scfi_barrel_merger.outputHitCollection,
        outputProtoClusterCollection="EcalBarrelScFiProtoClusters",
        splitCluster=False,
        minClusterCenterEdep=10.*MeV,
        localDistXZ=[30*mm, 30*mm])
    algs.append(scfi_barrel_cl)

    scfi_barrel_clreco = RecoCoG("scfi_barrel_clreco",
        inputProtoClusterCollection=scfi_barrel_cl.outputProtoClusterCollection,
        outputClusterCollection="EcalBarrelScFiClusters",
        logWeightBase=6.2)
    algs.append(scfi_barrel_clreco)

# Central Barrel Hcal
cb_hcal_daq = dict(
         dynamicRangeADC=50.*MeV,
         capacityADC=32768,
         pedestalMean=400,
         pedestalSigma=10)

cb_hcal_digi = CalHitDigi("cb_hcal_digi",
         inputHitCollection="HcalBarrelHits",
         outputHitCollection="HcalBarrelHitsDigi",
         **cb_hcal_daq)
algs.append(cb_hcal_digi)

cb_hcal_reco = CalHitReco("cb_hcal_reco",
        inputHitCollection=cb_hcal_digi.outputHitCollection,
        outputHitCollection="HcalBarrelHitsReco",
        thresholdFactor=5.0,
        readoutClass="HcalBarrelHits",
        layerField="layer",
        sectorField="module",
        samplingFraction=cb_hcal_sf,
        **cb_hcal_daq)
algs.append(cb_hcal_reco)

cb_hcal_merger = CalHitsMerger("cb_hcal_merger",
        inputHitCollection=cb_hcal_reco.outputHitCollection,
        outputHitCollection="HcalBarrelHitsRecoXY",
        readoutClass="HcalBarrelHits",
        fields=["layer", "slice"],
        fieldRefNumbers=[1, 0])
algs.append(cb_hcal_merger)

cb_hcal_cl = IslandCluster("cb_hcal_cl",
        inputHitCollection=cb_hcal_merger.outputHitCollection,
        outputProtoClusterCollection="HcalBarrelProtoClusters",
        splitCluster=False,
        minClusterCenterEdep=30.*MeV,
        localDistXY=[15.*cm, 15.*cm])
algs.append(cb_hcal_cl)

cb_hcal_clreco = RecoCoG("cb_hcal_clreco",
        inputProtoClusterCollection=cb_hcal_cl.outputProtoClusterCollection,
        outputClusterCollection="HcalBarrelClusters",
        logWeightBase=6.2)
algs.append(cb_hcal_clreco)

# Hcal Hadron Endcap
ci_hcal_daq = dict(
         dynamicRangeADC=50.*MeV,
         capacityADC=32768,
         pedestalMean=400,
         pedestalSigma=10)

ci_hcal_digi = CalHitDigi("ci_hcal_digi",
         inputHitCollection="HcalEndcapPHits",
         outputHitCollection="HcalEndcapPHitsDigi",
         **ci_hcal_daq)
algs.append(ci_hcal_digi)

ci_hcal_reco = CalHitReco("ci_hcal_reco",
        inputHitCollection=ci_hcal_digi.outputHitCollection,
        outputHitCollection="HcalEndcapPHitsReco",
        thresholdFactor=5.0,
        samplingFraction=ci_hcal_sf,
        **ci_hcal_daq)
algs.append(ci_hcal_reco)

ci_hcal_merger = CalHitsMerger("ci_hcal_merger",
        inputHitCollection=ci_hcal_reco.outputHitCollection,
        outputHitCollection="HcalEndcapPHitsRecoXY",
        readoutClass="HcalEndcapPHits",
        fields=["layer", "slice"],
        fieldRefNumbers=[1, 0])
algs.append(ci_hcal_merger)

ci_hcal_cl = IslandCluster("ci_hcal_cl",
        inputHitCollection=ci_hcal_merger.outputHitCollection,
        outputProtoClusterCollection="HcalEndcapPProtoClusters",
        splitCluster=False,
        minClusterCenterEdep=30.*MeV,
        localDistXY=[15.*cm, 15.*cm])
algs.append(ci_hcal_cl)

ci_hcal_clreco = RecoCoG("ci_hcal_clreco",
        inputProtoClusterCollection=ci_hcal_cl.outputProtoClusterCollection,
        outputClusterCollection="HcalEndcapPClusters",
        logWeightBase=6.2)
algs.append(ci_hcal_clreco)

# Hcal Electron Endcap
ce_hcal_daq = dict(
        dynamicRangeADC=50.*MeV,
        capacityADC=32768,
        pedestalMean=400,
        pedestalSigma=10)

ce_hcal_digi = CalHitDigi("ce_hcal_digi",
        inputHitCollection="HcalEndcapNHits",
        outputHitCollection="HcalEndcapNHitsDigi",
        **ce_hcal_daq)
algs.append(ce_hcal_digi)

ce_hcal_reco = CalHitReco("ce_hcal_reco",
        inputHitCollection=ce_hcal_digi.outputHitCollection,
        outputHitCollection="HcalEndcapNHitsReco",
        thresholdFactor=5.0,
        samplingFraction=ce_hcal_sf,
        **ce_hcal_daq)
algs.append(ce_hcal_reco)

ce_hcal_merger = CalHitsMerger("ce_hcal_merger",
        inputHitCollection=ce_hcal_reco.outputHitCollection,
        outputHitCollection="HcalEndcapNHitsRecoXY",
        readoutClass="HcalEndcapNHits",
        fields=["layer", "slice"],
        fieldRefNumbers=[1, 0])
algs.append(ce_hcal_merger)

ce_hcal_cl = IslandCluster("ce_hcal_cl",
        inputHitCollection=ce_hcal_merger.outputHitCollection,
        outputProtoClusterCollection="HcalEndcapNProtoClusters",
        splitCluster=False,
        minClusterCenterEdep=30.*MeV,
        localDistXY=[15.*cm, 15.*cm])
algs.append(ce_hcal_cl)

ce_hcal_clreco = RecoCoG("ce_hcal_clreco",
        inputProtoClusterCollection=ce_hcal_cl.outputProtoClusterCollection,
        outputClusterCollection="HcalEndcapNClusters",
        logWeightBase=6.2)
algs.append(ce_hcal_clreco)

# output
podout = PodioOutput("out", filename=output_rec)
podout.outputCommands = ['drop *',
        'keep MCParticles',
        'keep *Digi',
        'keep *Reco*',
        'keep *Cluster*',
        'keep *Layers']
algs.append(podout)

ApplicationMgr(
    TopAlg = algs,
    EvtSel = 'NONE',
    EvtMax = n_events,
    ExtSvc = [podioevent],
    OutputLevel=WARNING
)
