/*
 *  An algorithm to group readout hits from a calorimeter
 *  Energy is summed
 *
 *  Author: Chao Peng (ANL), 03/31/2021
 */
#include <bitset>
#include <algorithm>
#include <unordered_map>

#include "Gaudi/Property.h"
#include "GaudiAlg/GaudiAlgorithm.h"
#include "GaudiKernel/ToolHandle.h"
#include "GaudiAlg/Transformer.h"
#include "GaudiAlg/GaudiTool.h"
#include "GaudiKernel/RndmGenerators.h"
#include "GaudiKernel/PhysicalConstants.h"

#include "DDRec/CellIDPositionConverter.h"
#include "DDRec/SurfaceManager.h"
#include "DDRec/Surface.h"
#include "DDSegmentation/BitFieldCoder.h"

#include "fmt/ranges.h"
#include "fmt/format.h"

// FCCSW
#include "JugBase/DataHandle.h"
#include "JugBase/IGeoSvc.h"

// Event Model related classes
#include "eicd/CalorimeterHitCollection.h"

using namespace Gaudi::Units;

namespace Jug::Reco {

class CalorimeterHitsMerger : public GaudiAlgorithm {
public:
    Gaudi::Property<std::string>                m_geoSvcName{this, "geoServiceName", "GeoSvc"};
    Gaudi::Property<std::string>                m_readout{this, "readoutClass", "EcalBarrelHits"};
    // field names to generate id mask, the hits will be grouped by masking the field
    Gaudi::Property<std::vector<std::string>>   u_fields{this, "fields", {"layer"}};
    // reference field numbers to locate position for each merged hits group
    Gaudi::Property<std::vector<int>>           u_refs{this, "fieldRefNumbers", {}};
    DataHandle<eic::CalorimeterHitCollection>
        m_inputHitCollection{"inputHitCollection", Gaudi::DataHandle::Reader, this};
    DataHandle<eic::CalorimeterHitCollection>
        m_outputHitCollection{"outputHitCollection", Gaudi::DataHandle::Writer, this};

    SmartIF<IGeoSvc> m_geoSvc;
    uint64_t id_mask, ref_mask;

    // ill-formed: using GaudiAlgorithm::GaudiAlgorithm;
    CalorimeterHitsMerger(const std::string& name, ISvcLocator* svcLoc)
        : GaudiAlgorithm(name, svcLoc)
    {
        declareProperty("inputHitCollection",       m_inputHitCollection,       "");
        declareProperty("outputHitCollection",      m_outputHitCollection,    "");
    }

    StatusCode initialize() override
    {
        if (GaudiAlgorithm::initialize().isFailure()) {
            return StatusCode::FAILURE;
        }

        m_geoSvc = service(m_geoSvcName);
        if (!m_geoSvc) {
            error() << "Unable to locate Geometry Service. "
                    << "Make sure you have GeoSvc and SimSvc in the right order in the configuration." << endmsg;
            return StatusCode::FAILURE;
        }

        try {
            auto id_desc = m_geoSvc->detector()->readout(m_readout).idSpec();
            id_mask = 0;
            std::vector<std::pair<std::string, int>> ref_fields;
            for (size_t i = 0; i < u_fields.size(); ++i) {
                id_mask |= id_desc.field(u_fields[i])->mask();
                // use the same ref number if the length of two vectors do not much
                // or default number (1) if emepty vector provied
                int ref = u_refs.empty() ? 1 : (i < u_refs.size() ? u_refs[i] : u_refs.value().back());
                ref_fields.push_back({u_fields[i], ref});
            }
            ref_mask = id_desc.encode(ref_fields);
            // debug() << fmt::format("Referece id mask for the fields {:#064b}", ref_mask) << endmsg;
        } catch (...) {
            error() << "Failed to load ID decoder for " << m_readout << endmsg;
            return StatusCode::FAILURE;
        }
        id_mask = ~id_mask;
        info() << fmt::format("ID mask in {:s}: {:#064b}", m_readout, id_mask)
               << endmsg;
        return StatusCode::SUCCESS;
    }

    StatusCode execute() override
    {
        // input collections
	    const auto &hits = *m_inputHitCollection.get();
        // Create output collections
        auto &mhits = *m_outputHitCollection.createAndPut();

        // dd4hep decoders
        auto poscon = m_geoSvc->cellIDPositionConverter();
        auto volman = m_geoSvc->detector()->volumeManager();

        // sum energies that has the same id
        std::unordered_map<long long, size_t> merge_map;
        for (auto &h : hits) {
            auto id = (h.cellID() & id_mask);
            // debug() << h.cellID() << " - " << std::bitset<64>(h.cellID()) << endmsg;
            auto it = merge_map.find(id);
            if (it == merge_map.end()) {
                merge_map[id] = mhits.size();
                auto ahit = h.clone();
                // use the reference field position
                int ref_id = id | ref_mask;
                // global positions
                auto gpos = poscon->position(ref_id);
                // local positions
                auto alignment = volman.lookupDetector(ref_id).nominal();
                auto pos = alignment.worldToLocal(dd4hep::Position(gpos.x(), gpos.y(), gpos.z()));
                ahit.position({gpos.x()/dd4hep::mm, gpos.y()/dd4hep::mm, gpos.z()/dd4hep::mm});
                ahit.local({pos.x()/dd4hep::mm, pos.y()/dd4hep::mm, pos.z()/dd4hep::mm});
                mhits.push_back(ahit);
                debug() << mhits[mhits.size() - 1].cellID() << " - " << std::bitset<64>(id) << endmsg;
            } else {
                mhits[it->second].energy(mhits[it->second].energy() + h.energy());
            }
        }

        /*
        for (auto &h : mhits) {
            debug() << h.cellID() << ": " << h.energy() << endmsg;
        }
        */

        debug() << "Size before = " << hits.size() << ", after = " << mhits.size() << endmsg;

        return StatusCode::SUCCESS;
    }


}; // class CalorimeterHitsMerger

DECLARE_COMPONENT(CalorimeterHitsMerger)

} // namespace Jug::Reco
