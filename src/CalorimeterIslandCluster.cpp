// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2022 Sylvester Joosten, Chao, Chao Peng, Wouter Deconinck, Jihee Kim, Whitney Armstrong

/*
 *  Island Clustering Algorithm for Calorimeter Blocks
 *  1. group all the adjacent modules
 *  2. split the groups between their local maxima with the energy deposit above <minClusterCenterEdep>
 *
 *  Author: Chao Peng (ANL), 09/27/2020
 *  References:
 *      https://cds.cern.ch/record/687345/files/note01_034.pdf
 *      https://www.jlab.org/primex/weekly_meetings/primexII/slides_2012_01_20/island_algorithm.pdf
 */
#include <algorithm>
#include <functional>
#include <unordered_map>

#include "fmt/format.h"

#include "DDRec/CellIDPositionConverter.h"
#include "DDRec/Surface.h"
#include "DDRec/SurfaceManager.h"

// Event Model related classes
#include "eicd/CalorimeterHitCollection.h"
#include "eicd/ClusterCollection.h"
#include "eicd/ProtoClusterCollection.h"
#include "eicd/Vector2f.h"
#include "eicd/Vector3f.h"
#include "eicd/vector_utils.h"

// Custom Headers
#include "AlgoHeaders.h"

#include <CL/sycl.hpp>


using namespace my_units;

using CaloHit = eicd::CalorimeterHit;
using CaloHitCollection = eicd::CalorimeterHitCollection;
using ProtoClusterCollection = eicd::ProtoClusterCollection;

namespace {

// helper functions to get distance between hits
static eicd::Vector2f localDistXY(const CaloHit& h1, const CaloHit& h2) {
  const auto delta = h1.getLocal() - h2.getLocal();
  return {delta.x, delta.y};
}

static eicd::Vector2f localDistXZ(const CaloHit& h1, const CaloHit& h2) {
  const auto delta = h1.getLocal() - h2.getLocal();
  return {delta.x, delta.z};
}

static eicd::Vector2f localDistYZ(const CaloHit& h1, const CaloHit& h2) {
  const auto delta = h1.getLocal() - h2.getLocal();
  return {delta.y, delta.z};
}

static eicd::Vector2f dimScaledLocalDistXY(const CaloHit& h1, const CaloHit& h2) {
  const auto delta = h1.getLocal() - h2.getLocal();
  const auto dimsum = h1.getDimension() + h2.getDimension();
  return {2 * delta.x / dimsum.x, 2 * delta.y / dimsum.y};
}

static eicd::Vector2f globalDistRPhi(const CaloHit& h1, const CaloHit& h2) {
  using vector_type = decltype(eicd::Vector2f::a);
  return {
    static_cast<vector_type>(
      eicd::magnitude(h1.getPosition()) - eicd::magnitude(h2.getPosition())
    ),
    static_cast<vector_type>(
      eicd::angleAzimuthal(h1.getPosition()) - eicd::angleAzimuthal(h2.getPosition())
    )
  };
}

static eicd::Vector2f globalDistEtaPhi(const CaloHit& h1,
                                       const CaloHit& h2) {
  using vector_type = decltype(eicd::Vector2f::a);
  return {
    static_cast<vector_type>(
      eicd::eta(h1.getPosition()) - eicd::eta(h2.getPosition())
    ),
    static_cast<vector_type>(
      eicd::angleAzimuthal(h1.getPosition()) - eicd::angleAzimuthal(h2.getPosition())
    )
  };
}

// name: {method, units}
static std::map<std::string,
                std::tuple<std::function<eicd::Vector2f(const CaloHit&, const CaloHit&)>, std::vector<double>>>
    distMethods{
        {"localDistXY", {localDistXY, {mm, mm}}},        {"localDistXZ", {localDistXZ, {mm, mm}}},
        {"localDistYZ", {localDistYZ, {mm, mm}}},        {"dimScaledLocalDistXY", {dimScaledLocalDistXY, {1., 1.}}},
        {"globalDistRPhi", {globalDistRPhi, {mm, rad}}}, {"globalDistEtaPhi", {globalDistEtaPhi, {1., rad}}},
    };


struct async_err_handler {
  void operator()(sycl::exception_list elist){
    for(auto e : elist){
      std::rethrow_exception(e);
    }
  }
};


// SYCL Initializations
sycl::default_selector device_selector;
sycl::queue queue(device_selector, async_err_handler{});

} // namespace

namespace Jug::Reco {

/**
 *  Island Clustering Algorithm for Calorimeter Blocks.
 *
 *  1. group all the adjacent modules
 *  2. split the groups between their local maxima with the energy deposit above <minClusterCenterEdep>
 *
 *  References:
 *      https://cds.cern.ch/record/687345/files/note01_034.pdf
 *      https://www.jlab.org/primex/weekly_meetings/primexII/slides_2012_01_20/island_algorithm.pdf
 *
 * \ingroup reco
 */

  int CalorimeterIslandCluster::initialize() {

    // unitless conversion, keep consistency with juggler internal units (GeV, mm, ns, rad)
    minClusterHitEdep    = p.m_minClusterHitEdep / GeV;
    minClusterCenterEdep = p.m_minClusterCenterEdep / GeV;
    sectorDist           = p.m_sectorDist / mm;

    hitsDist = localDistXY;

    /* std::cout << "Running on "
              << queue.get_device().get_info<cl::sycl::info::device::name>()
              << "\n"; */

    return 0;
  }

  ProtoClusterCollection CalorimeterIslandCluster::execute() {
    // input collections
    const auto& hits = m_inputHitCollection;

    // Create output collections
    ProtoClusterCollection proto;

    // group neighboring hits
    std::vector<std::vector<std::pair<uint32_t, CaloHit>>> groups;

    parallel_group(groups, hits);

    for (auto& group : groups) {
      if (group.empty()) {
        continue;
      }

      for(auto p : group){
        std::cout << "Group: " << p.first << " Hit: " << p.second.id() << "\n";
      }

      auto maxima = find_maxima(group, !m_splitCluster);
      split_group(group, maxima, proto);
      if (p.is_debug) {
        debug() << "hits in a group: " << group.size() << ", "
                << "local maxima: " << maxima.size() << endmsg;
      }
    }

    std::cout << "Total Number of groups for this event: " << groups.size() << "\n";
    return proto;
  }

  // helper function to group hits
  inline bool CalorimeterIslandCluster::is_neighbour(const CaloHit& h1, const CaloHit& h2) const {
    // in the same sector
    if (h1.getSector() == h2.getSector()) {
      auto dist = hitsDist(h1, h2);
      return (dist.a <= neighbourDist[0]) && (dist.b <= neighbourDist[1]);
      // different sector, local coordinates do not work, using global coordinates
    } else {
      // sector may have rotation (barrel), so z is included
      return (eicd::magnitude(h1.getPosition() - h2.getPosition()) <= sectorDist);
    }
  }

  SYCL_EXTERNAL inline bool is_neighbour( const sycl::accessor<sycl::float3,1,sycl::access::mode::read>& lpos,
                                          const sycl::accessor<sycl::float3,1,sycl::access::mode::read>& gpos,
                                          const sycl::accessor<int,1,sycl::access::mode::read>& sectors,
                                          const sycl::accessor<double,1,sycl::access::mode::read>& secDist,
                                          const sycl::accessor<double,1,sycl::access::mode::read>& neighDist,
                                          sycl::id<1> hit1, int hit2){
    
    if(sectors[hit1] == sectors[hit2]) {
        // localXY
        auto delta = lpos[hit1] - lpos[hit2];
        return (delta.x() <= neighDist[0] && delta.y() <= neighDist[1]);
      } else {
        auto delta = gpos[hit1] - gpos[hit2];
        return (sycl::length(delta) <= secDist[0]);
      }
  }

  // Parallel Grouping Algorithm
  void CalorimeterIslandCluster::parallel_group(std::vector<std::vector<std::pair<uint32_t, CaloHit>>>& groups,
                const CaloHitCollection& hits) const {

    // Corner Cases
    if(hits.size() <= 0) return;

    // Host memory
    //
    // Get location data from hits
    std::vector<int32_t> sectors;
    std::vector<float> energy;
    std::vector<sycl::vec<float, 3>> lpos, gpos;

    // Neighbour Indices
    std::vector<int> nidx (hits.size());

    // Can't filter out hits here as index numbers will not match
    for(size_t i = 0; i < hits.size(); i++) {
      lpos.emplace_back(hits[i].getLocal().x, hits[i].getLocal().y, hits[i].getLocal().z);
      gpos.emplace_back(hits[i].getPosition().x, hits[i].getPosition().y, hits[i].getPosition().z);
      energy.emplace_back(hits[i].getEnergy());
      sectors.emplace_back(hits[i].getSector());
    }

    {
      // Device memory
      sycl::buffer<int, 1> nidx_buf (nidx.data(), sycl::range<1>(nidx.size()));
      sycl::buffer<double, 1> minClusterHitEdep_buf (&minClusterHitEdep, sycl::range<1>(1));
      sycl::buffer<float, 1> energy_buf (energy.data(), sycl::range<1>(energy.size()));

      sycl::buffer<sycl::float3,1> lpos_buf (lpos.data(), sycl::range<1>(lpos.size()));
      sycl::buffer<sycl::float3,1> gpos_buf (gpos.data(), sycl::range<1>(lpos.size()));

      sycl::buffer<int,1> sectors_buf (sectors.data(), sycl::range<1>(sectors.size()));
      sycl::buffer<double, 1> sectorDist_buf (&sectorDist, sycl::range<1>(1));
      sycl::buffer<double, 1> neighbourDist_buf (neighbourDist.data(), sycl::range<1>(neighbourDist.size()));

      try {
            // Initalize Neighbour Indices
            queue.submit([&](sycl::handler& h) {
              auto nidx_acc = nidx_buf.get_access<sycl::access::mode::write>(h);
              h.parallel_for(sycl::range<1>(hits.size()), [=](sycl::id<1> idx) {
                  nidx_acc[idx] = idx;
              });
            });

            /**
             * @brief Assign current vertex id (idx) to neighbour if idx < id
             * held at neighbour index. Emulates sequential assignment of 
             * clusters by DFS in parallel.
             */
            queue.submit([&](sycl::handler& h) {
              auto nidx_acc = nidx_buf.get_access<sycl::access::mode::atomic>(h);
              auto minClusterHitEdep_acc = minClusterHitEdep_buf.get_access<sycl::access::mode::read>(h);
              auto energy_acc = energy_buf.get_access<sycl::access::mode::read>(h);

              auto lpos_acc = lpos_buf.get_access<sycl::access::mode::read>(h);
              auto gpos_acc = gpos_buf.get_access<sycl::access::mode::read>(h);

              auto sectors_acc = sectors_buf.get_access<sycl::access::mode::read>(h);
              auto sectorDist_acc = sectorDist_buf.get_access<sycl::access::mode::read>(h);
              auto neighbourDist_acc = neighbourDist_buf.get_access<sycl::access::mode::read>(h);

              sycl::stream dbg (1024,1024,h);
              h.parallel_for(sycl::range<1>(hits.size()), [=](sycl::id<1> idx) {
                // not a qualified hit
                if(energy_acc[idx] < minClusterHitEdep_acc[0]){
                  return;
                }

                for(size_t i = 0; i < nidx_acc.size(); i++) {

                  if(energy_acc[i] < minClusterHitEdep_acc[0]) continue;

                  if(!Jug::Reco::is_neighbour(lpos_acc,gpos_acc,sectors_acc,sectorDist_acc,neighbourDist_acc,idx,i)){
                    continue;
                  }
                  
                  // Atomic exchange of min element between current neighbour index nidx_acc[i] and idx
                  nidx_acc[i].fetch_min(idx);

                }

              });
            }).wait_and_throw();
      } catch(sycl::exception e) {
        std::cerr << "Caught SYCL Exception: " << e.what() << "\n";
      }

    } // Sync Device and Host memory

    // Emplace index array into groups for further processing
/*     std::cout << "Grouping Results are:\n";
    for(int i : nidx) {
      std::cout << i << " ";
    }
    std::cout << "\n"; */

    std::unordered_map<int, std::vector<std::pair<uint32_t, CaloHit>>> gr;
    for(size_t i = 0; i < hits.size(); i++) {
      gr[nidx[i]].emplace_back(i, hits[i]);
    }
    for(auto i : gr) {
      groups.emplace_back(i.second);
    }

  }

  // grouping function with Depth-First Search
  void CalorimeterIslandCluster::dfs_group(std::vector<std::pair<uint32_t, CaloHit>>& group, int idx,
                 const CaloHitCollection& hits, std::vector<bool>& visits) const {
    // not a qualified hit to particpate clustering, stop here
    if (hits[idx].getEnergy() < minClusterHitEdep) {
      visits[idx] = true;
      //std::cout << "hit " << idx << "has energy less than threshold\n";
      return;
    }

    group.emplace_back(idx, hits[idx]);
    visits[idx] = true;
    for (size_t i = 0; i < hits.size(); ++i) {
      if(is_neighbour(hits[idx], hits[i])){
        //std::cout << idx << " and " << i << " are neighbours\n";
      }
      if (visits[i] || !is_neighbour(hits[idx], hits[i])) {
        continue;
      }
      dfs_group(group, i, hits, visits);
    }
  }

  // find local maxima that above a certain threshold
  std::vector<CaloHit>
  CalorimeterIslandCluster::find_maxima(const std::vector<std::pair<uint32_t, CaloHit>>& group,
              bool global = false) const {

    std::vector<CaloHit> maxima;

    if (group.empty()) {
      return maxima;
    }

    if (global) {
      int mpos = 0;
      for (size_t i = 0; i < group.size(); ++i) {
        if (group[mpos].second.getEnergy() < group[i].second.getEnergy()) {
          mpos = i;
        }
      }
      if (group[mpos].second.getEnergy() >= minClusterCenterEdep) {
        maxima.push_back(group[mpos].second);
      }
      return maxima;
    }

    // Prep Host memory for device offload
    std::vector<float> energy;
    std::vector<uint8_t> max_idx (group.size());

    // Get location data from hits
    std::vector<int32_t> sectors;
    std::vector<sycl::vec<float, 3>> lpos, gpos;

    for (const auto& [idx, hit] : group){
      lpos.emplace_back(hit.getLocal().x, hit.getLocal().y, hit.getLocal().z);
      gpos.emplace_back(hit.getPosition().x, hit.getPosition().y, hit.getPosition().z);
      energy.push_back(hit.getEnergy());
      sectors.push_back(hit.getSector());
    }

    // Device memory
    {

      sycl::buffer<float, 1> energy_buf (energy.data(), sycl::range<1>(energy.size()));
      sycl::buffer<uint8_t, 1> max_idx_buf (max_idx.data(), sycl::range<1>(max_idx.size()));
      sycl::buffer<double, 1> minClusterCenterEdep_buf (&minClusterCenterEdep, sycl::range<1>(1));

      sycl::buffer<sycl::float3,1> lpos_buf (lpos.data(), sycl::range<1>(lpos.size()));
      sycl::buffer<sycl::float3,1> gpos_buf (gpos.data(), sycl::range<1>(lpos.size()));

      sycl::buffer<int32_t,1> sectors_buf (sectors.data(), sycl::range<1>(sectors.size()));
      sycl::buffer<double, 1> sectorDist_buf (&sectorDist, sycl::range<1>(1));
      sycl::buffer<double, 1> neighbourDist_buf (neighbourDist.data(), sycl::range<1>(neighbourDist.size()));
      
      try {
        queue.submit([&](sycl::handler& h){
          auto energy_acc = energy_buf.get_access<sycl::access::mode::read>(h);
          auto max_idx_acc = max_idx_buf.get_access<sycl::access::mode::write>(h);
          auto minClusterCenterEdep_acc = minClusterCenterEdep_buf.get_access<sycl::access::mode::read>(h);

          auto lpos_acc = lpos_buf.get_access<sycl::access::mode::read>(h);
          auto gpos_acc = gpos_buf.get_access<sycl::access::mode::read>(h);

          auto sectors_acc = sectors_buf.get_access<sycl::access::mode::read>(h);
          auto sectorDist_acc = sectorDist_buf.get_access<sycl::access::mode::read>(h);
          auto neighbourDist_acc = neighbourDist_buf.get_access<sycl::access::mode::read>(h);

          sycl::stream dbg (1024,1024, h);
          h.parallel_for(sycl::range<1>(group.size()), [=](sycl::id<1> idx){
            if(energy_acc[idx] < minClusterCenterEdep_acc[0]){
              return;
            }
            
            bool is_max = true;

            for(int i = 0; i < energy_acc.size(); i++){
              if(idx == i){
                continue;
              }

              if(energy_acc[i] > energy_acc[idx]){
                if(Jug::Reco::is_neighbour(lpos_acc,gpos_acc,sectors_acc,sectorDist_acc,neighbourDist_acc,idx,i)){
                      is_max = false;
                      break;
                  }
                      
              }
            }

            if(is_max){
              max_idx_acc[idx] = 1;
            }
            
          });
        }).wait_and_throw();

      } catch(sycl::exception e) {
        std::cerr << "Caught SYLC Exception: " << e.what() << "\n";
      }

    } // Sync Device memory to Host

    // Convert maxima index array to hit vector for further processing
    for(size_t i = 0; i < max_idx.size(); i++){
      if(max_idx[i] == 1){
        std::cout << "Found Maxima at: " << group[i].first << "\n";
        maxima.push_back(group[i].second);
      }
    }

    return maxima;

    for (const auto& [idx, hit] : group) {
      // not a qualified center
      if (hit.getEnergy() < minClusterCenterEdep) {
        continue;
      }

      bool maximum = true;
      for (const auto& [idx2, hit2] : group) {
        if (hit == hit2) {
          continue;
        }

        if (is_neighbour(hit, hit2) && hit2.getEnergy() > hit.getEnergy()) {
          maximum = false;
          break;
        }
      }

      if (maximum) {
        std::cout << "Found Maxima at: " << idx << "\n";
        maxima.push_back(hit);
      }
    }

    return maxima;
  }

  // helper function
  inline void CalorimeterIslandCluster::vec_normalize(std::vector<double>& vals) {
    double total = 0.;
    for (auto& val : vals) {
      total += val;
    }
    for (auto& val : vals) {
      val /= total;
    }
  }

  // split a group of hits according to the local maxima
  void CalorimeterIslandCluster::split_group(const std::vector<std::pair<uint32_t, CaloHit>>& group, const std::vector<CaloHit>& maxima,
                   ProtoClusterCollection& proto) const {
    // special cases
    if (maxima.empty()) {
      if (p.is_verbose) {
        verbose() << "No maxima found, not building any clusters" << endmsg;
      }
      return;
    } else if (maxima.size() == 1) {
      eicd::MutableProtoCluster pcl;
      for (auto& [idx, hit] : group) {
        pcl.addToHits(hit);
        pcl.addToWeights(1.);
      }
      proto.push_back(pcl);
      if (p.is_verbose) {
        verbose() << "A single maximum found, added one ProtoCluster" << endmsg;
      }
      return;
    }

    // split between maxima
    // TODO, here we can implement iterations with profile, or even ML for better splits
    std::vector<double> weights(maxima.size(), 1.);
    std::vector<eicd::MutableProtoCluster> pcls;
    for (size_t k = 0; k < maxima.size(); ++k) {
      pcls.emplace_back();
    }

    size_t i = 0;
    for (const auto& [idx, hit] : group) {
      size_t j = 0;
      // calculate weights for local maxima
      for (const auto& chit : maxima) {
        double dist_ref = chit.getDimension().x;
        double energy   = chit.getEnergy();
        double dist     = eicd::magnitude(hitsDist(chit, hit));
        weights[j]      = std::exp(-dist / dist_ref) * energy;
        j += 1;
      }

      // normalize weights
      vec_normalize(weights);

      // ignore small weights
      for (auto& w : weights) {
        if (w < 0.02) {
          w = 0;
        }
      }
      vec_normalize(weights);

      // split energy between local maxima
      for (size_t k = 0; k < maxima.size(); ++k) {
        double weight = weights[k];
        if (weight <= 1e-6) {
          continue;
        }
        pcls[k].addToHits(hit);
        pcls[k].addToWeights(weight);
      }
      i += 1;
    }
    for (auto& pcl : pcls) {
      proto.push_back(pcl);
    }
    if (p.is_verbose) {
      verbose() << "Multiple (" << maxima.size() << ") maxima found, added a ProtoClusters for each maximum" << endmsg;
    }
  }

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)

} // namespace Jug::Reco
