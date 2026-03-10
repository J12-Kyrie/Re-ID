# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Re-ID (Person Re-Identification) module for NVIDIA DRIVE AGX Orin platform. Implements dual-DLA parallel inference pipeline for real-time person re-identification across multiple cameras using TensorRT and DriveWorks SDK.

## Build System

**CMake-based build** (requires DriveWorks SDK 5.20):

```bash
cd reid
mkdir build && cd build
cmake .. -DDRIVEWORKS_DIR=/path/to/driveworks-5.20
make
```

**Dependencies:**
- DriveWorks SDK 5.20 (default path: `../../driveworks-5.20` relative to `reid/`)
- TensorRT 8.6.13+
- CUDA Toolkit
- C++11

## Architecture

### Core Pipeline (Triple-Queue Design)

The system uses a **static memory pool with slot-based indexing** to eliminate dynamic allocation:

1. **Free Queue** → Available `slot_id` pool
2. **Pending Queue** → ROIs ready for DLA inference
3. **Completion Queue** → Inference results ready for association

Each `slot_id` maps to pre-allocated GPU tensors (input: 1×3×256×128 FP16, output: 1×256 FP16).

### Component Hierarchy

```
ReIdManager (dual-DLA orchestrator)
├── ReIdWorker0 (DLA Core 0)
│   ├── ReIdConditioner (dwDataConditioner for ROI preprocessing)
│   └── ReIdEngine (TensorRT runtime)
└── ReIdWorker1 (DLA Core 1)
    ├── ReIdConditioner
    └── ReIdEngine
```

**Key Classes:**
- `ReIdManager`: Dual-worker coordinator, round-robin ROI distribution
- `ReIdWorker`: Single DLA worker (conditioner + engine + CUDA stream)
- `ReIdEngine`: TensorRT inference wrapper (loads `.engine`, manages I/O tensors)
- `ReIdConditioner`: DriveWorks ROI preprocessing (VIC hardware acceleration)
- `ReIdConfigLoader`: JSON config parser

### Data Structures

**`dwRect`**: Bounding box with `(x, y)` as **top-left corner**, `width`, `height`

**Core structs** (from `develop.md`):
- `ReIDRoiRequest`: Input from YOLO (image handle, bbox, camera_id, track_id)
- `ReIDPreparedInput`: Slot metadata + tensor handles
- `ReIDResult`: Inference output (slot_id, embedding pointer)

## Configuration

**`reid/config/reid_config.json`**:
```json
{
  "enginePathDla0": "reid_resnet50_dla0.engine",
  "enginePathDla1": "reid_resnet50_dla1.engine",
  "engineHasPreprocess": false
}
```

Paths are relative to `ReId/` directory. Set `engineHasPreprocess: true` if engine includes mean/std normalization.

## DLA Performance (Orin Platform)

**Dual-DLA throughput**: ~89-90 embeddings/s (Batch=1, p99 latency ~22.4ms)
**Single DLA**: ~46 embeddings/s (p99 ~21.7ms)
**GPU (Ampere)**: ~1041 embeddings/s (p99 ~1.0ms) - for comparison

**Best practices:**
- Use **Batch=1** (逐帧推理) - Batch≥4 causes severe GPU fallback
- Build separate engines per DLA core (`--useDLACore=0/1`)
- Enable `--allowGPUFallback` for unsupported layers (Shape/Gather/Shuffle)

## Engine Building

```bash
# DLA 0
trtexec --onnx=resnet50_market1501_aicity156.onnx \
  --saveEngine=reid_resnet50_dla0.engine \
  --useDLACore=0 --allowGPUFallback \
  --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw

# DLA 1 (同理)
trtexec --onnx=resnet50_market1501_aicity156.onnx \
  --saveEngine=reid_resnet50_dla1.engine \
  --useDLACore=1 --allowGPUFallback \
  --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
```

## API Usage

**Initialization:**
```cpp
dwContextHandle_t ctx;
ReIdManagerConfig config;
loadReIdConfig("reid/config/reid_config.json", &config);

ReIdManager manager;
manager.init(ctx, &config);
```

**Inference:**
```cpp
dwImageHandle_t image;
dwRect rois[N];  // N bounding boxes
std::vector<std::vector<float>> embeddings;

manager.process(image, rois, N, &embeddings);
// embeddings[i] is 256-dim float vector for rois[i]
```

## Thread Safety

- `ReIdManager::process()` spawns threads internally for dual-DLA parallelism
- Each `ReIdWorker` owns its CUDA stream - no cross-worker synchronization needed
- For production: replace `std::thread` with thread pool + triple-queue (see `develop2.md`)

## Important Notes

- **Coordinate system**: Image origin at top-left, x-right, y-down
- **Memory alignment**: DLA requires 256-byte aligned tensors
- **Stream synchronization**: Ensure VIC preprocessing completes before pushing to Pending Queue (use `cudaEvent_t`)
- **Error handling**: Check `isInitialized()` before calling `process()`
