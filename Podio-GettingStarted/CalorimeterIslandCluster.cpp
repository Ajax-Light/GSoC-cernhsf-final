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
    for(int i = 0; i < (int) hits.size(); i++){
      if(hits[i].getEnergy() < minClusterHitEdep){
        continue;
      }else if(is_neighbour(hits[idx], hits[i])){
        std::cout << i << " ";
        nlist.push_back(i);
      }
    }
    std::cout << "\n";
    return nlist;
  }

  // Functor to find neighbours within SYCL code
  struct neighbour {

    neighbour(sycl::accessor<int32_t,1,sycl::access::mode::read>& sectors,
              sycl::accessor<float,1,sycl::access::mode::read>& lx,
              sycl::accessor<float,1,sycl::access::mode::read>& ly,
              sycl::accessor<float,1,sycl::access::mode::read>& lz,
              sycl::accessor<float,1,sycl::access::mode::read>& gx,
              sycl::accessor<float,1,sycl::access::mode::read>& gy,
              sycl::accessor<float,1,sycl::access::mode::read>& gz,
              sycl::accessor<double,1,sycl::access::mode::read>& nDist,
              sycl::accessor<double,1,sycl::access::mode::read>& secDist):
              sectors(sectors), lx(lx), ly(ly), lz(lz), gx(gx), gy(gy), gz(gz), nDist(nDist), secDist(secDist) {}

    // is_neighbour
    bool operator() (sycl::id<1> hit1, sycl::id<1> hit2) const {
      if(sectors[hit1] == sectors[hit2]) {
        // localXY
        float a = lx[hit1] - lx[hit2];
        float b = ly[hit1] - lx[hit2];
        return (a <= nDist[0] && b <= nDist[1]);
      } else {
        float x = gx[hit1] - gx[hit2];
        float y = gy[hit1] - gy[hit2];
        float z = gz[hit1] - gz[hit2];
        double magnitude = sycl::sqrt(x*x+y*y+z*z);
        return (magnitude <= secDist[0]);
      }
    }

    private:
      sycl::accessor<int32_t,1,sycl::access::mode::read> sectors;
      sycl::accessor<float,1,sycl::access::mode::read> lx,ly,lz,gx,gy,gz;
      sycl::accessor<double,1,sycl::access::mode::read> nDist,secDist;

  };

  // Parallel Grouping Algorithm (ECL-CC)
  void CalorimeterIslandCluster::parallel_group(std::vector<std::vector<std::pair<uint32_t, CaloHit>>>& groups,
                const CaloHitCollection& hits) const {

    // Corner Cases
    if(hits.size() <= 0) return;

    // Host memory
    //
    // Get location data from hits
    std::vector<int32_t> sectors;
    std::vector<float> lx,ly,lz,gx,gy,gz;

    for(size_t i = 0; i < hits.size(); i++){
      if(hits[i].getEnergy() < minClusterHitEdep) continue;

      sectors.emplace_back(hits[i].getSector());
      gx.emplace_back(hits[i].getPosition().x);
      lx.emplace_back(hits[i].getLocal().x);
      gy.emplace_back(hits[i].getPosition().y);
      ly.emplace_back(hits[i].getLocal().y);
      gz.emplace_back(hits[i].getPosition().z);
      lz.emplace_back(hits[i].getLocal().z);
    }


    // Disjoint-Set Union or Union-Find Structure
    std::vector<int> nidx(sectors.size());

    {
      // Device memory
      sycl::buffer<int, 1> nidx_buf (nidx.data(), sycl::range<1>(nidx.size()));

      sycl::buffer<float, 1> lx_buf (lx.data(), sycl::range<1>(lx.size()));
      sycl::buffer<float, 1> ly_buf (ly.data(), sycl::range<1>(ly.size()));
      sycl::buffer<float, 1> lz_buf (lz.data(), sycl::range<1>(lz.size()));
      sycl::buffer<float, 1> gx_buf (gx.data(), sycl::range<1>(gx.size()));
      sycl::buffer<float, 1> gy_buf (gy.data(), sycl::range<1>(gy.size()));
      sycl::buffer<float, 1> gz_buf (gz.data(), sycl::range<1>(gz.size()));

      sycl::buffer<int32_t,1> sectors_buf (sectors.data(), sycl::range<1>(sectors.size()));
      sycl::buffer<double, 1> sectorDist_buf (&sectorDist, sycl::range<1>(1));
      sycl::buffer<double, 1> neighbourDist_buf (neighbourDist.data(), sycl::range<1>(neighbourDist.size()));

      try{
            // Initalize DSU Structure
            queue.submit([&](sycl::handler& h){
              auto nidx_acc = nidx_buf.get_access<sycl::access::mode::write>(h);
              h.parallel_for(sycl::range<1>(nidx.size()), [=](sycl::id<1> idx){
                nidx_acc[idx] = idx;
              });
            });

            // Hooking (Union)
            queue.submit([&](sycl::handler& h){
              auto nidx_acc = nidx_buf.get_access<sycl::access::mode::atomic>(h);

              auto lx_acc = lx_buf.get_access<sycl::access::mode::read>(h);
              auto ly_acc = ly_buf.get_access<sycl::access::mode::read>(h);
              auto lz_acc = lz_buf.get_access<sycl::access::mode::read>(h);
              auto gx_acc = gx_buf.get_access<sycl::access::mode::read>(h);
              auto gy_acc = gy_buf.get_access<sycl::access::mode::read>(h);
              auto gz_acc = gz_buf.get_access<sycl::access::mode::read>(h);

              auto sectors_acc = sectors_buf.get_access<sycl::access::mode::read>(h);
              auto sectorDist_acc = sectorDist_buf.get_access<sycl::access::mode::read>(h);
              auto neighbourDist_acc = neighbourDist_buf.get_access<sycl::access::mode::read>(h);

              neighbour neigh(sectors_acc,lx_acc,ly_acc,lz_acc,gx_acc,gy_acc,gz_acc,neighbourDist_acc,sectorDist_acc);

              sycl::stream dbg (1024,1024,h);
              h.parallel_for(sycl::range<1>(nidx.size()), [=](sycl::id<1> idx){

                for(size_t i = 0; i < nidx_acc.size(); i++){
                  dbg << "idx: "<< idx[1] << " i: " << i << " : " << neigh(idx[0], i) << "\n";
                  if(!neigh(idx[0], i))
                    continue;
                  
                  const int neigh = i;
                  bool repeat = false;
                  do {
                    repeat = false;
                    int ret = nidx_acc[neigh].load();
                    if(ret > idx)
                      nidx_acc[neigh].compare_exchange_strong(ret, idx);
                    if(ret != nidx_acc[neigh].load()){
                      repeat = true;
                    }
                  }while(repeat);
                }

              });
            }).wait_and_throw();
      }catch(sycl::exception e){
        std::cerr << "Caught SYCL Exception: " << e.what() << "\n";
      }

    } // Sync Device and Host memory

    std::cout << "Grouping Results are:\n";
    for(int i : nidx){
      std::cout << i << " ";
    }
    std::cout << "\n";

    std::unordered_map<int, std::vector<std::pair<uint32_t, CaloHit>>> gr;
    for(size_t i = 0; i < hits.size(); i++){
      gr[nidx[i]].emplace_back(i, hits[i]);
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
