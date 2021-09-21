#include <algorithm>
#include <cmath>
#include <fmt/format.h>

#include "Gaudi/Algorithm.h"
#include "GaudiAlg/GaudiTool.h"
#include "GaudiAlg/Producer.h"
#include "GaudiAlg/Transformer.h"
#include "GaudiKernel/RndmGenerators.h"

#include "JugBase/DataHandle.h"
#include "JugBase/UniqueID.h"

// Event Model related classes
#include "eicd/TrackerHitCollection.h"
#include "eicd/ReconstructedParticleCollection.h"

namespace Jug::Reco {

class FarForwardParticles : public GaudiAlgorithm, AlgorithmIDMixin<> {
DataHandle<eic::TrackerHitCollection> m_inputHitCollection{"FarForwardTrackerHits", Gaudi::DataHandle::Reader, this};
        DataHandle<eic::ReconstructedParticleCollection> m_outputParticles{"outputParticles", Gaudi::DataHandle::Writer, this};
        
        //----- Define constants here ------
        
        Gaudi::Property<double> local_x_offset_station_1{this, "localXOffsetSta1", -833.3878326};
        Gaudi::Property<double> local_x_offset_station_2{this, "localXOffsetSta2", -924.342804};
        Gaudi::Property<double> local_x_slope_offset{this, "localXSlopeOffset", -0.00622147};
        Gaudi::Property<double> local_y_slope_offset{this, "localYSlopeOffset", -0.0451035};
        Gaudi::Property<double> crossingAngle{this, "crossingAngle", -0.025};
        Gaudi::Property<double> NOM_MOMENTUM{this, "beamMomentum", 275.0};
        Gaudi::Property<double> base{this, "detSeparationZ", 2000.0};
        
        
        
    public:
        FarForwardParticles(const std::string& name, ISvcLocator* svcLoc)
        : GaudiAlgorithm(name, svcLoc), AlgorithmIDMixin(name, info()) {
            declareProperty("inputCollection", m_inputHitCollection, "FarForwardTrackerHits");
            declareProperty("outputCollection", m_outputParticles, "ReconstructedParticles");
        }
        
        StatusCode initialize() override {
            if (GaudiAlgorithm::initialize().isFailure())
                return StatusCode::FAILURE;
            
            return StatusCode::SUCCESS;
        }
        
        StatusCode execute() override {
            const eic::TrackerHitCollection* rawhits = m_inputHitCollection.get();
            auto& rc       = *(m_outputParticles.createAndPut());
            
            
            //for (const auto& part : mc) {
            //    if (part.genStatus() > 1) {
            //        if (msgLevel(MSG::DEBUG)) {
            //            debug() << "ignoring particle with genStatus = " << part.genStatus() << endmsg;
            //        }
            //        continue;
            //    }
                
                //---- begin Roman Pot Reconstruction code ----
            
                double aXRP[2][2] = {{2.102403743, 29.11067626}, {0.186640381, 0.192604619}};
                double aYRP[2][2] = {{0.0000159900, 3.94082098}, {0.0000079946, -0.1402995}};
                
                int eventReset = 0; //counter for IDing at least one hit per layer
                double hitx[100];
                double hity[100];
                double hitz[100];
            
                double aXRPinv[2][2];
                double aYRPinv[2][2];
            
            
                double det = aXRP[0][0]*aXRP[1][1] - aXRP[0][1]*aXRP[1][0];
            
                aXRPinv[0][0] =  aXRP[1][1]/det;
                aXRPinv[0][1] = -aXRP[0][1]/det;
                aXRPinv[1][0] = -aXRP[1][0]/det;
                aXRPinv[1][1] =  aXRP[0][0]/det;
            
                det = aYRP[0][0]*aYRP[1][1] - aYRP[0][1]*aYRP[1][0];
                aYRPinv[0][0] =  aYRP[1][1]/det;
                aYRPinv[0][1] = -aYRP[0][1]/det;
                aYRPinv[1][0] = -aYRP[1][0]/det;
                aYRPinv[1][1] =  aYRP[0][0]/det;
            
                for (const auto& h : *rawhits) {
                    
                    // The actual hit position:
                    auto pos0 = h.position();
                    //auto mom0 = h.momentum;
                    //auto pidCode = h.g4ID;
                    auto eDep = h.edep();
                    
                    if(eDep < 0.00001){continue;}
                    
                    if(eventReset < 2) {hitx[eventReset] = pos0.x - local_x_offset_station_1;}
                    else{hitx[eventReset] = pos0.x - local_x_offset_station_2;}
                    
                    hity[eventReset] = pos0.y;
                    hitz[eventReset] = pos0.z;
                    
                    eventReset++;
                }
                
                //NB:
                //This is a "dumb" algorithm - I am just checking the basic thing works with a simple single-proton test.
                //Will need to update and modify for generic # of hits for more complicated final-states.

                if(eventReset == 4){ 
                    
                    
                    //extract hit, subtract orbit offset – this is to get the hits in the coordinate system of the orbit trajectory
                    double XL[2], YL[2];
                    
                    XL[0] = hitx[0]; YL[0] = hity[0];
                    XL[1] = hitx[2]; YL[1] = hity[2];
                    
                    double Xrp[2], Xip[2]; Xrp[0] = XL[1]; Xrp[1] = (1000*(XL[1] - XL[0])/(hitz[3] - hitz[0])) - local_x_slope_offset; //- _SX0RP_;
                    double Yrp[2], Yip[2]; Yrp[0] = YL[1]; Yrp[1] = (1000*(YL[1] - YL[0])/(hitz[3] - hitz[0])) - local_y_slope_offset; //- _SY0RP_;
                    
                    Xip[0] = Xip[1] = 0.0;
                    Yip[0] = Yip[1] = 0.0;
                    
                    //use the hit information and calculated slope at the RP + the transfer matrix inverse to calculate the Polar Angle and deltaP at the IP
                    
                    for(unsigned i0=0; i0<2; i0++){
                        for(unsigned i1=0; i1<2; i1++){
                            Xip[i0] += aXRPinv[i0][i1]*Xrp[i1];
                            Yip[i0] += aYRPinv[i0][i1]*Yrp[i1];
                        }
                    }
                    
                    //convert polar angles to radians
                    double rsx = Xip[1]/1000., rsy = Yip[1]/1000.;
                    
                    //calculate momentum magnitude from measured deltaP – using thin lens optics.
                    double p = NOM_MOMENTUM*(1 + 0.01*Xip[0]);
                    double norm = std::sqrt(1.0 + rsx*rsx + rsy*rsy);
                    
                    double prec[3];
                    prec[0] = p*rsx/norm;
                    prec[1] = p*rsy/norm;
                    prec[2] = p/norm;
                    
                    //double pT_reco = std::sqrt(prec[0]*prec[0] + prec[1]*prec[1]);
                    float p_reco = std::sqrt(prec[0]*prec[0] + prec[1]*prec[1] + prec[2]*prec[2]);
                    
                    //----- end RP reconstruction code ------
                    
                    
                    eic::VectorXYZ recoMom(prec[0], prec[1], prec[2]);
                    eic::VectorXYZ primVtx(0, 0, 0);
                    eic::Direction thetaPhi(recoMom.theta(), recoMom.phi());
                    eic::Weight wgt(1);
                    
                    //ReconstructedParticle (eic::Index ID, eic::VectorXYZ p, eic::VectorXYZ v, float time, std::int32_t pid, std::int16_t status, std::int16_t charge, eic::Weight weight, eic::Direction direction, float momentum, float energy, float mass)
                    
                    //eic::ReconstructedParticle rec_part{{part.ID(), algorithmID()},
                    //mom3s,
                    //{part.vs().x, part.vs().y, part.vs().z},
                    //static_cast<float>(part.time()),
                    //part.pdgID(),
                    //0,
                    //static_cast<int16_t>(part.charge()),
                    //1.,
                    //{mom3s.theta(), mom3s.phi()},
                    //static_cast<float>(moms),
                    //static_cast<float>(Es),
                    //static_cast<float>(part.mass())};

                    eic::ReconstructedParticle rpTrack{{-1,-1}, recoMom, primVtx, 0.0, 1, 0, 1, 1., {recoMom.theta(), recoMom.phi()}, p_reco, p_reco, 0.938};
                    rc->push_back(rpTrack);
                    
                
                } //end enough hits for matrix reco
            
            return StatusCode::SUCCESS;
        }
    
    
        
    };
    
    DECLARE_COMPONENT(FarForwardParticles)
    
} // namespace Jug::Reco

