#ifndef GLOBAL_ASSOCIATION_H
#define GLOBAL_ASSOCIATION_H

#include "ReIDTypes.h"
#include "ReIDResourceManager.h"
#include <unordered_map>
#include <vector>
#include <thread>
#include <atomic>

namespace reid {

class GlobalGallery {
public:
    GlobalGallery();
    ~GlobalGallery() = default;

    void getCandidates(uint64_t current_ts,
                      std::vector<GlobalTrackMeta*>& active_cands,
                      std::vector<GlobalTrackMeta*>& lost_cands);

    uint32_t createNewTrack(const float* feat, uint64_t ts,
                           uint8_t cam_id, uint32_t local_id);

    void updateTrack(uint32_t gid, const float* new_feat, uint64_t ts,
                    uint8_t cam_id, uint32_t local_id);

    void purgeStaleTracks(uint64_t current_ts);

    size_t getActiveCount() const;
    size_t getTotalCount() const { return tracks_.size(); }

private:
    std::unordered_map<uint32_t, GlobalTrackMeta> tracks_;
    uint32_t next_global_id_;

    static constexpr uint64_t ACTIVE_THRESHOLD_US = 2000000;
    static constexpr uint64_t MAX_TTL_US = 30000000;
    static constexpr float EMA_MOMENTUM = 0.9f;
};

class AssociationMatcher {
public:
    AssociationMatcher() = default;
    ~AssociationMatcher() = default;

    MatchResult match(const float* target_feat,
                     const std::vector<GlobalTrackMeta*>& active_cands,
                     const std::vector<GlobalTrackMeta*>& lost_cands);

private:
    float computeCosineSimilarity(const float* vecA, const float* vecB);
    MatchResult performGreedy(const float* target_feat,
                             const std::vector<GlobalTrackMeta*>& candidates);

    static constexpr float COSINE_SIM_THRESHOLD = 0.65f;
};

class GlobalAssociation {
public:
    GlobalAssociation();
    ~GlobalAssociation();

    GlobalAssociation(const GlobalAssociation&) = delete;
    GlobalAssociation& operator=(const GlobalAssociation&) = delete;

    bool initialize(ReIDResourceManager* pool_manager);
    void release();

    void start();
    void stop();

    bool isInitialized() const { return initialized_; }

private:
    void consumerThreadLoop();
    void processSlot(uint32_t slot_id);

    GlobalGallery gallery_;
    AssociationMatcher matcher_;
    ReIDResourceManager* pool_manager_;

    std::thread consumer_thread_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> frame_counter_;
    bool initialized_;
};

}

#endif
