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

/*     std::vector<bool> visits(hits.size(), false);
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
    }  */

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

  // Get Neighbour hit indices
  std::vector<int> CalorimeterIslandCluster::get_neighbours(const CaloHitCollection& hits, int idx) const{
    std::vector<int> nlist;
    std::cout << "Index: " << idx << " Neighbours: ";
    for(int i = idx; i < (int) hits.size(); i++){
      if(hits[i].getEnergy() < minClusterHitEdep){
        std::cout << "hit " << idx << "has energy less than threshold\n";
        continue;
      }else if(is_neighbour(hits[idx], hits[i])){
        std::cout << i << " ";
        nlist.push_back(i);
      }
    }
    std::cout << "\n";
    return nlist;
  }

  // Find representative / root node of a component with Intermediate-Pointer Jumping (Path Compression)
  SYCL_EXTERNAL inline int representative(const int idx, sycl::accessor<int,1,sycl::access::mode::read_write> dsu) {
    int par = dsu[idx];
    if(par != idx){
      int next, prev = idx;
      while(par > (next = dsu[par])){
        dsu[prev] = next;   // Intermediate Path Compression / pointer jumping
        prev = par;
        par = next;
      }
    }
    return par;
  }

  // Parallel Grouping Algorithm (ECL-CC)
  void CalorimeterIslandCluster::parallel_group(std::vector<std::vector<std::pair<uint32_t, CaloHit>>>& groups,
                const CaloHitCollection& hits) const {
    
    // Corner Cases
    if(hits.size() <= 0) return;

    // Convert Hits data to CSR Format for Device Offload
    std::vector<int> nidx {0}, nlist;
    int curr_size = 0;
    for(size_t i = 0; i < hits.size(); i++){
      if(hits[i].getEnergy() < minClusterHitEdep) continue;

      auto neigh = get_neighbours(hits, i);
      curr_size += (int)neigh.size();
      nidx.push_back(curr_size);
      for(int t : neigh){
        nlist.push_back(t);
      }
    }

    // Host memory
    //
    // Disjoint-Set Union or Union-Find Structure
    std::vector<int> dsu(nidx.size() - 1);

    {
      // Device memory
      sycl::buffer<int, 1> dsu_buf (dsu.data(), sycl::range<1>(dsu.size()));
      sycl::buffer<int, 1> nidx_buf (nidx.data(), sycl::range<1>(nidx.size()));
      sycl::buffer<int, 1> nlist_buf (nlist.data(), sycl::range<1>(nlist.size()));
      

      try{
            // Initalize DSU Structure
            queue.submit([&](sycl::handler& h){
              auto dsu_acc = dsu_buf.get_access<sycl::access::mode::write>(h);
              h.parallel_for(sycl::range<1>(dsu.size()), [=](sycl::id<1> idx){
                dsu_acc[idx] = idx;
              });
            });

            // Hooking (Union)
            queue.submit([&](sycl::handler& h){
              auto dsu_acc = dsu_buf.get_access<sycl::access::mode::read_write>(h);
              auto nidx_acc = nidx_buf.get_access<sycl::access::mode::read>(h);
              auto nlist_acc = nlist_buf.get_access<sycl::access::mode::read>(h);
              sycl::stream dbg (1024,1024,h);
              h.parallel_for(sycl::range<1>(dsu.size()), [=](sycl::id<1> idx){
                dbg << "Started Hooking at idx: " << idx << "\n";
                const int begin = nidx_acc[idx];
                const int end = nidx_acc[idx+1];
                
                int v_rep = representative(idx, dsu_acc);
                dbg << "Representative of " << idx << " is " << v_rep << "\n";

                for(int i = begin; i < end; i++){
                  
                  const int neigh = nlist_acc[i];
                  dbg << "Current Neighbour: " << neigh << "\n";
                  if(true){    // Process edges only in one direction - doesn't work in our case ?? FIX
                    int u_rep = representative(neigh, dsu_acc);
                    bool repeat = false;
                    do {
                      repeat = false;
                      // Check if they belong to different components and update respresentative to smaller one
                      if(v_rep != u_rep){
                        if(v_rep < u_rep){
                          int ret = u_rep;
                          auto atm = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                      sycl::memory_scope::device,
                                                      sycl::access::address_space::ext_intel_global_device_space>(dsu_acc[u_rep]);
                          atm.compare_exchange_strong(ret, v_rep); // Handle changes to representative by other threads
                          if(ret != u_rep){
                            u_rep = ret;
                            repeat = true;
                          }
                        }else{
                          int ret = v_rep;
                          auto atm = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                      sycl::memory_scope::device,
                                                      sycl::access::address_space::ext_intel_global_device_space>(dsu_acc[v_rep]);
                          repeat = atm.compare_exchange_strong(ret, u_rep); // Handle changes to representative by other threads
                          if(ret != v_rep){
                            v_rep = ret;
                            repeat = true;
                          }
                        }
                      }
                    } while(repeat);
                  }
                }

              });
            }).wait_and_throw();
      }catch(sycl::exception e){
        std::cerr << "Caught SYCL Exception: " << e.what() << "\n";
      }

    } // Sync Device and Host memory

    for(int i : dsu){
      std::cout << i << " ";
    }
    std::cout << "\n";

    std::unordered_map<int, std::vector<std::pair<uint32_t, CaloHit>>> gr;
    for(size_t i = 0; i < hits.size(); i++){
      gr[dsu[i]].emplace_back(i, hits[i]);
    }
    for(auto i : gr){
      groups.emplace_back(i.second);
    }

  }

  // grouping function with Depth-First Search
  void CalorimeterIslandCluster::dfs_group(std::vector<std::pair<uint32_t, CaloHit>>& group, int idx,
                 const CaloHitCollection& hits, std::vector<bool>& visits) const {
    // not a qualified hit to particpate clustering, stop here
    if (hits[idx].getEnergy() < minClusterHitEdep) {
      visits[idx] = true;
      std::cout << "hit " << idx << "has energy less than threshold\n";
      return;
    }

    group.emplace_back(idx, hits[idx]);
    visits[idx] = true;
    for (size_t i = 0; i < hits.size(); ++i) {
      if(is_neighbour(hits[idx], hits[i])){
        std::cout << idx << " and " << i << " are neighbours\n";
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

/*     std::vector<bool> visits(hits.size(), false);
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
    }  */

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

  // Get Neighbour hit indices
  std::vector<int> CalorimeterIslandCluster::get_neighbours(const CaloHitCollection& hits, int idx) const{
    std::vector<int> nlist;
    std::cout << "Index: " << idx << " Neighbours: ";
    for(int i = idx; i < (int) hits.size(); i++){
      if(hits[i].getEnergy() < minClusterHitEdep){
        std::cout << "hit " << idx << "has energy less than threshold\n";
        continue;
      }else if(is_neighbour(hits[idx], hits[i])){
        std::cout << i << " ";
        nlist.push_back(i);
      }
    }
    std::cout << "\n";
    return nlist;
  }

  // Find representative / root node of a component with Intermediate-Pointer Jumping (Path Compression)
  SYCL_EXTERNAL inline int representative(const int idx, sycl::accessor<int,1,sycl::access::mode::read_write> dsu) {
    int par = dsu[idx];
    if(par != idx){
      int next, prev = idx;
      while(par > (next = dsu[par])){
        dsu[prev] = next;   // Intermediate Path Compression / pointer jumping
        prev = par;
        par = next;
      }
    }
    return par;
  }

  // Parallel Grouping Algorithm (ECL-CC)
  void CalorimeterIslandCluster::parallel_group(std::vector<std::vector<std::pair<uint32_t, CaloHit>>>& groups,
                const CaloHitCollection& hits) const {
    
    // Corner Cases
    if(hits.size() <= 0) return;

    // Convert Hits data to CSR Format for Device Offload
    std::vector<int> nidx {0}, nlist;
    int curr_size = 0;
    for(size_t i = 0; i < hits.size(); i++){
      if(hits[i].getEnergy() < minClusterHitEdep) continue;

      auto neigh = get_neighbours(hits, i);
      curr_size += (int)neigh.size();
      nidx.push_back(curr_size);
      for(int t : neigh){
        nlist.push_back(t);
      }
    }

    // Host memory
    //
    // Disjoint-Set Union or Union-Find Structure
    std::vector<int> dsu(nidx.size() - 1);

    {
      // Device memory
      sycl::buffer<int, 1> dsu_buf (dsu.data(), sycl::range<1>(dsu.size()));
      sycl::buffer<int, 1> nidx_buf (nidx.data(), sycl::range<1>(nidx.size()));
      sycl::buffer<int, 1> nlist_buf (nlist.data(), sycl::range<1>(nlist.size()));
      

      try{
            // Initalize DSU Structure
            queue.submit([&](sycl::handler& h){
              auto dsu_acc = dsu_buf.get_access<sycl::access::mode::write>(h);
              h.parallel_for(sycl::range<1>(dsu.size()), [=](sycl::id<1> idx){
                dsu_acc[idx] = idx;
              });
            });

            // Hooking (Union)
            queue.submit([&](sycl::handler& h){
              auto dsu_acc = dsu_buf.get_access<sycl::access::mode::read_write>(h);
              auto nidx_acc = nidx_buf.get_access<sycl::access::mode::read>(h);
              auto nlist_acc = nlist_buf.get_access<sycl::access::mode::read>(h);
              sycl::stream dbg (1024,1024,h);
              h.parallel_for(sycl::range<1>(dsu.size()), [=](sycl::id<1> idx){
                dbg << "Started Hooking at idx: " << idx << "\n";
                const int begin = nidx_acc[idx];
                const int end = nidx_acc[idx+1];
                
                int v_rep = representative(idx, dsu_acc);
                dbg << "Representative of " << idx << " is " << v_rep << "\n";

                for(int i = begin; i < end; i++){
                  
                  const int neigh = nlist_acc[i];
                  dbg << "Current Neighbour: " << neigh << "\n";
                  if(true){    // Process edges only in one direction - doesn't work in our case ?? FIX
                    int u_rep = representative(neigh, dsu_acc);
                    bool repeat = false;
                    do {
                      repeat = false;
                      // Check if they belong to different components and update respresentative to smaller one
                      if(v_rep != u_rep){
                        if(v_rep < u_rep){
                          int ret = u_rep;
                          auto atm = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                      sycl::memory_scope::device,
                                                      sycl::access::address_space::ext_intel_global_device_space>(dsu_acc[u_rep]);
                          atm.compare_exchange_strong(ret, v_rep); // Handle changes to representative by other threads
                          if(ret != u_rep){
                            u_rep = ret;
                            repeat = true;
                          }
                        }else{
                          int ret = v_rep;
                          auto atm = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                      sycl::memory_scope::device,
                                                      sycl::access::address_space::ext_intel_global_device_space>(dsu_acc[v_rep]);
                          repeat = atm.compare_exchange_strong(ret, u_rep); // Handle changes to representative by other threads
                          if(ret != v_rep){
                            v_rep = ret;
                            repeat = true;
                          }
                        }
                      }
                    } while(repeat);
                  }
                }

              });
            }).wait_and_throw();
      }catch(sycl::exception e){
        std::cerr << "Caught SYCL Exception: " << e.what() << "\n";
      }

    } // Sync Device and Host memory

    for(int i : dsu){
      std::cout << i << " ";
    }
    std::cout << "\n";

    std::unordered_map<int, std::vector<std::pair<uint32_t, CaloHit>>> gr;
    for(size_t i = 0; i < hits.size(); i++){
      gr[dsu[i]].emplace_back(i, hits[i]);
    }
    for(auto i : gr){
      groups.emplace_back(i.second);
    }

  }

  // grouping function with Depth-First Search
  void CalorimeterIslandCluster::dfs_group(std::vector<std::pair<uint32_t, CaloHit>>& group, int idx,
                 const CaloHitCollection& hits, std::vector<bool>& visits) const {
    // not a qualified hit to particpate clustering, stop here
    if (hits[idx].getEnergy() < minClusterHitEdep) {
      visits[idx] = true;
      std::cout << "hit " << idx << "has energy less than threshold\n";
      return;
    }

    group.emplace_back(idx, hits[idx]);
    visits[idx] = true;
    for (size_t i = 0; i < hits.size(); ++i) {
      if(is_neighbour(hits[idx], hits[i])){
        std::cout << idx << " and " << i << " are neighbours\n";
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
