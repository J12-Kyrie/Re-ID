#include "GlobalAssociation.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace reid {

GlobalGallery::GlobalGallery()
    : next_global_id_(1) {
}

void GlobalGallery::getCandidates(uint64_t current_ts,
                                  std::vector<GlobalTrackMeta*>& active_cands,
                                  std::vector<GlobalTrackMeta*>& lost_cands) {
    active_cands.clear();
    lost_cands.clear();

    for (auto& pair : tracks_) {
        if (pair.second.state == TrackState::ACTIVE) {
            active_cands.push_back(&pair.second);
        } else if (pair.second.state == TrackState::LOST) {
            lost_cands.push_back(&pair.second);
        }
    }
}

uint32_t GlobalGallery::createNewTrack(const float* feat, uint64_t ts,
                                       uint8_t cam_id, uint32_t local_id) {
    uint32_t gid = next_global_id_++;

    GlobalTrackMeta meta;
    meta.global_id = gid;
    meta.state = TrackState::ACTIVE;
    meta.first_seen_ts = ts;
    meta.last_seen_ts = ts;
    meta.last_camera_id = cam_id;
    meta.last_local_id = local_id;
    meta.hit_streak = 1;

    std::memcpy(meta.centroid_embedding, feat, REID_FEAT_DIM * sizeof(float));

    tracks_[gid] = meta;
    return gid;
}

void GlobalGallery::updateTrack(uint32_t gid, const float* new_feat, uint64_t ts,
                                uint8_t cam_id, uint32_t local_id) {
    auto it = tracks_.find(gid);
    if (it == tracks_.end()) {
        return;
    }

    GlobalTrackMeta& track = it->second;
    track.last_seen_ts = ts;
    track.last_camera_id = cam_id;
    track.last_local_id = local_id;
    track.state = TrackState::ACTIVE;
    track.hit_streak++;

    float norm_sq = 0.0f;
    for (size_t i = 0; i < REID_FEAT_DIM; ++i) {
        track.centroid_embedding[i] =
            EMA_MOMENTUM * track.centroid_embedding[i] +
            (1.0f - EMA_MOMENTUM) * new_feat[i];
        norm_sq += track.centroid_embedding[i] * track.centroid_embedding[i];
    }

    float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-6f);
    for (size_t i = 0; i < REID_FEAT_DIM; ++i) {
        track.centroid_embedding[i] *= inv_norm;
    }
}

void GlobalGallery::purgeStaleTracks(uint64_t current_ts) {
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        uint64_t dt = current_ts - it->second.last_seen_ts;

        if (dt > MAX_TTL_US) {
            it = tracks_.erase(it);
        } else {
            if (dt > ACTIVE_THRESHOLD_US && it->second.state == TrackState::ACTIVE) {
                it->second.state = TrackState::LOST;
                it->second.hit_streak = 0;
            }
            ++it;
        }
    }
}

size_t GlobalGallery::getActiveCount() const {
    return std::count_if(tracks_.begin(), tracks_.end(),
        [](const std::pair<uint32_t, GlobalTrackMeta>& p) {
            return p.second.state == TrackState::ACTIVE;
        });
}

float AssociationMatcher::computeCosineSimilarity(const float* vecA, const float* vecB) {
    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (size_t i = 0; i < REID_FEAT_DIM; ++i) {
        dot += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }

    return dot / (std::sqrt(normA) * std::sqrt(normB) + 1e-6f);
}

MatchResult AssociationMatcher::performGreedy(const float* target_feat,
                                              const std::vector<GlobalTrackMeta*>& candidates) {
    MatchResult best_match;
    best_match.matched = false;
    best_match.global_id = 0;
    best_match.similarity = -1.0f;

    for (const auto* cand : candidates) {
        float sim = computeCosineSimilarity(target_feat, cand->centroid_embedding);

        if (sim > COSINE_SIM_THRESHOLD && sim > best_match.similarity) {
            best_match.matched = true;
            best_match.global_id = cand->global_id;
            best_match.similarity = sim;
        }
    }

    return best_match;
}

MatchResult AssociationMatcher::match(const float* target_feat,
                                     const std::vector<GlobalTrackMeta*>& active_cands,
                                     const std::vector<GlobalTrackMeta*>& lost_cands) {
    MatchResult res = performGreedy(target_feat, active_cands);
    if (res.matched) {
        return res;
    }

    return performGreedy(target_feat, lost_cands);
}

GlobalAssociation::GlobalAssociation()
    : pool_manager_(nullptr)
    , running_(false)
    , frame_counter_(0)
    , initialized_(false) {
}

GlobalAssociation::~GlobalAssociation() {
    release();
}

bool GlobalAssociation::initialize(ReIDResourceManager* pool_manager) {
    if (initialized_) {
        return true;
    }

    if (!pool_manager) {
        std::cerr << "GlobalAssociation: Invalid pool manager" << std::endl;
        return false;
    }

    pool_manager_ = pool_manager;
    initialized_ = true;
    return true;
}

void GlobalAssociation::release() {
    if (!initialized_) {
        return;
    }

    stop();
    initialized_ = false;
}

void GlobalAssociation::start() {
    if (!initialized_ || running_.load()) {
        return;
    }

    running_.store(true);
    consumer_thread_ = std::thread(&GlobalAssociation::consumerThreadLoop, this);

    std::cout << "GlobalAssociation: Started consumer thread" << std::endl;
}

void GlobalAssociation::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    if (consumer_thread_.joinable()) {
        consumer_thread_.join();
    }

    std::cout << "GlobalAssociation: Stopped consumer thread" << std::endl;
}

void GlobalAssociation::consumerThreadLoop() {
    constexpr uint32_t kBatchSize = 16;
    uint32_t slot_ids[kBatchSize];

    while (running_.load()) {
        uint32_t slot_id;
        if (!pool_manager_->popCompletion(slot_id, 100)) {
            continue;
        }

        processSlot(slot_id);
    }
}

void GlobalAssociation::processSlot(uint32_t slot_id) {
    ReIDPreparedInput* meta = pool_manager_->getPreparedInput(slot_id);
    float* current_feat = pool_manager_->getOutputEmbeddingBuffer(slot_id);
    uint64_t current_ts = meta->timestamp_us;

    std::vector<GlobalTrackMeta*> active_cands, lost_cands;
    gallery_.getCandidates(current_ts, active_cands, lost_cands);

    MatchResult result = matcher_.match(current_feat, active_cands, lost_cands);

    uint32_t final_gid;
    if (result.matched) {
        gallery_.updateTrack(result.global_id, current_feat, current_ts,
                           meta->camera_id, meta->local_track_id);
        final_gid = result.global_id;
    } else {
        final_gid = gallery_.createNewTrack(current_feat, current_ts,
                                           meta->camera_id, meta->local_track_id);
    }

    pool_manager_->recycleSlot(slot_id);

    frame_counter_.fetch_add(1);
    if (frame_counter_.load() % 50 == 0) {
        gallery_.purgeStaleTracks(current_ts);
    }
}

}
