#ifndef REID_TYPES_H
#define REID_TYPES_H

#include <cstdint>
#include <atomic>

namespace reid {

// DriveWorks type definitions (mocked for compilation)
typedef void* dwImageHandle_t;
typedef void* dwContextHandle_t;
typedef void* dwDNNTensorHandle_t;
typedef uint64_t dwTime_t;

struct dwRect {
    int32_t x;       // Top-left X coordinate
    int32_t y;       // Top-left Y coordinate
    int32_t width;   // Width in pixels
    int32_t height;  // Height in pixels
};

// Slot state machine for memory pool management
enum class SlotState : uint8_t {
    FREE = 0,           // Available in Free Queue
    PREPROCESSING,      // VIC is processing
    PENDING,           // Ready for DLA inference
    INFERRING,         // DLA is processing
    COMPLETED          // Result ready, in Completion Queue
};

// Track lifecycle state for cross-camera association
enum class TrackState : uint8_t {
    ACTIVE = 0,         // Seen within last 2 seconds
    LOST = 1,           // Not seen for 2-30 seconds
    DEAD = 2            // Not seen for >30 seconds (ready for GC)
};

// Callback function type for image release notification
typedef void (*ReleaseCallback)(dwImageHandle_t image, void* user_data);

// Input request from YOLO worker
struct ReIDRequest {
    dwImageHandle_t image;              // Image handle (YOLO owns until callback)
    dwRect bbox;                        // Bounding box (x,y = top-left corner)
    uint32_t local_track_id;            // YOLO's per-camera track ID
    uint8_t camera_id;                  // Camera index [0-5]
    dwTime_t timestamp_us;              // Microsecond timestamp
    ReleaseCallback release_callback;   // Called when VIC completes
    void* user_data;                    // Opaque context pointer
};

// Output result for YOLO consumer
struct ReIDResult {
    uint32_t local_track_id;            // Original track ID from request
    uint8_t camera_id;                  // Camera ID from request
    dwTime_t timestamp_us;              // Timestamp from request
    dwRect source_bbox;                 // Original bounding box
    float embedding[256];               // 256-dim feature vector (FP32)
    uint32_t global_track_id;           // Cross-camera global ID (0 = not associated)
    float association_confidence;       // Association confidence [0.0-1.0]
};

// Internal preprocessed input metadata
struct ReIDPreparedInput {
    uint32_t slot_id;                   // Memory slot identifier
    uint8_t camera_id;
    uint32_t local_track_id;
    dwTime_t timestamp_us;
    dwRect source_bbox;

    ReleaseCallback release_callback;   // Image release callback
    void* user_data;                    // Opaque user context

    void* preprocessing_done_event;     // cudaEvent_t for sync
};

// Memory pool slot containing all tensor data
struct TensorSlot {
    // GPU memory pointers (256-byte aligned)
    void* d_input_ptr;                  // Input tensor: FP16 [1,3,256,128]
    void* d_output_ptr;                 // Output tensor: FP16 [1,256]
    float* h_embedding_ptr;             // CPU-accessible output: FP32 [256]

    // DriveWorks tensor handles
    dwDNNTensorHandle_t input_tensor;
    dwDNNTensorHandle_t output_tensor;

    // CUDA synchronization events
    void* preprocessing_done_event;     // cudaEvent_t
    void* inference_done_event;         // cudaEvent_t

    // Slot state management
    std::atomic<SlotState> state;
    uint32_t slot_id;

    // Preprocessed input metadata
    ReIDPreparedInput prepared_input;
};

// Global track metadata for cross-camera association
struct GlobalTrackMeta {
    uint32_t global_id;                 // Unique global track identifier
    TrackState state;                   // Current lifecycle state

    uint8_t last_camera_id;             // Last observed camera
    uint32_t last_local_id;             // Last observed local track ID
    uint64_t first_seen_ts;             // Creation timestamp
    uint64_t last_seen_ts;              // Last update timestamp

    uint32_t hit_streak;                // Consecutive match count

    // EMA-updated centroid feature (L2-normalized, NEON-aligned)
    alignas(32) float centroid_embedding[256];
};

// Association matching result
struct MatchResult {
    bool matched;                       // Whether a match was found
    uint32_t global_id;                 // Matched global track ID
    float similarity;                   // Cosine similarity score
};

// Pipeline performance statistics
struct PipelineStats {
    uint32_t free_slots;                // Available slots in Free Queue
    uint32_t pending_tasks;             // Tasks waiting for DLA
    uint32_t completed_tasks;           // Results ready for polling

    uint64_t total_submitted;           // Lifetime counter: submissions
    uint64_t total_completed;           // Lifetime counter: completions
    uint64_t total_dropped;             // Lifetime counter: drops (pool exhausted)

    float dla0_utilization;             // DLA 0 utilization [0.0-1.0]
    float dla1_utilization;             // DLA 1 utilization [0.0-1.0]
};

// Pipeline configuration structure
struct ReIDPipelineConfig {
    const char* engine_path_dla0;       // Path to DLA 0 TensorRT engine
    const char* engine_path_dla1;       // Path to DLA 1 TensorRT engine
    uint32_t pool_size;                 // Memory pool size (recommended: 128)
    bool enable_global_association;     // Enable cross-camera matching
};

// Constants
constexpr uint32_t INVALID_SLOT_ID = 0xFFFFFFFF;
constexpr size_t REID_FEAT_DIM = 256;              // Embedding dimension
constexpr size_t REID_INPUT_BYTES = 196608;        // Input tensor size (1×3×256×128×2 bytes)
constexpr size_t REID_OUTPUT_BYTES = 512;          // Output tensor size (1×256×2 bytes)
constexpr size_t MEMORY_ALIGNMENT = 256;           // DLA alignment requirement

}

#endif
