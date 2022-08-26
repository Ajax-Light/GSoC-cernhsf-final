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
#include <tuple>
#include <stack>
#include <array>

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

namespace helper {

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

    hitsDist = helper::localDistXY;

    std::cout << "Running on "
              << helper::queue.get_device().get_info<cl::sycl::info::device::name>()
              << "\n";

    return 0;
  }

  ProtoClusterCollection CalorimeterIslandCluster::execute() {
    // input collections
    const auto& hits = m_inputHitCollection;
    // Create output collections
    //auto& proto = m_outputProtoCollection.createAndPut();
    ProtoClusterCollection proto;

    // group neighboring hits
    std::vector<std::vector<std::pair<uint32_t, CaloHit>>> groups;

    std::vector<bool> visits(hits.size(), false);
    for (size_t i = 0; i < hits.size(); ++i) {
      if (p.is_debug) {
        const auto& hit = hits[i];
        std::cout << fmt::format("hit {:d}: energy = {:.4f} MeV, local = ({:.4f}, {:.4f}) mm, "
                               "global=({:.4f}, {:.4f}, {:.4f}) mm",
                               i, hit.getEnergy() * 1000., hit.getLocal().x, hit.getLocal().y, hit.getPosition().x,
                               hit.getPosition().y, hit.getPosition().z)
                << std::endl;
      }
      // already in a group
      if (visits[i]) {
        continue;
      }
      groups.emplace_back();
      // create a new group, and group all the neighboring hits
      dfs_group(groups.back(), i, hits, visits);
    }

    for (auto& group : groups) {
      if (group.empty()) {
        continue;
      }
      auto maxima = find_maxima(group, !m_splitCluster);
      split_group(group, maxima, proto);
      if (p.is_debug) {
        debug() << "hits in a group: " << group.size() << ", "
                << "local maxima: " << maxima.size() << endmsg;
      }
    }

    //std::cout << "Total Number of groups for this event: " << groups.size() << "\n";
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

class parallel_dfs {
  public:
    void operator()(){

    }
};

  // grouping function with Depth-First Search
  void CalorimeterIslandCluster::dfs_group(std::vector<std::pair<uint32_t, CaloHit>>& group, int index,
                 const CaloHitCollection& hits, std::vector<bool>& visits) const {
    // Alg Inputs: hits, visits, group, index
    // Alg Outputs: visits (for subsequent calls), group

    // Fork Hits
    CaloHitCollection h1, h2;
    for(int i = 0; i < hits.size()/2; i++){
      h1.push_back(hits[i]);
    }
    for(int i = hits.size()/2; i < hits.size(); i++){
      h2.push_back(hits[i]);
    }

    // Vector<bool> is stored as a bit_map, hence we need convert into array to construct a buffer
    bool* arr = new bool[visits.size()];
    std::copy(visits.begin(), visits.end(), arr);

    { // Create buffers for all inputs and outputs
    sycl::buffer<bool, 1> vis_buf (arr, sycl::range<1>(visits.size()));
    //sycl::buffer<std::vector<std::pair<uint32_t, CaloHit>>, 1> group_buf (group, sycl::range<1>(group.size()));
    //sycl::buffer<CaloHitCollection, 1> h1_buf (h1, sycl::range<1>(1));

    helper::queue.submit([&](sycl::handler& cgh){
      
      auto vis_acc = vis_buf.get_access<sycl::access::mode::read_write>(cgh);
      
      cgh.single_task([=](){
        std::stack <int> s;
        s.push(index);

        while(!s.empty()){
        uint32_t idx = s.top();
        s.pop();

        // Already Visited
        if(vis_acc[idx]){
          continue;
        }
        // not a qualified hit to particpate clustering, stop here
        if (hits[idx].getEnergy() < minClusterHitEdep) {
          vis_acc[idx] = true;
          continue;
        }

        group.emplace_back(idx, hits[idx]);
        vis_acc[idx] = true;
        for (size_t i = 0; i < hits.size()/2; ++i) {
          if (vis_acc[i] || !is_neighbour(hits[idx], hits[i])) {
            continue;
          }
          
          s.push(i);
        }
      }
      });

    });

    try{
      helper::queue.wait_and_throw();
    } catch(std::exception e){
      std::cerr << "Caught SYCL Exception: " << e.what() << "\n";
    }  
    
    } // Buffer destructor called, memory is synced back to host

    // Copy from bool array to visits for next call

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
        maxima.push_back(hit);
      }
    }

    return maxima;
  }

  // helper function - Parallelization not needed
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
