存储项	建议位置	内存类型	理由
推理输出 (Raw Output)	Output Tensor Pool	GPU/DLA 可访问的对齐内存	仅用于接收 DLA 导出的原始 512 维特征，用完即释放。
环形队列 (Feature Pool)	GlobalTrackMeta (System RAM)	std::vector<float> 或预分配数组	存储的是经过 L2 归一化后的“洁净”特征。放在系统内存即可，因为匹配逻辑（Association）在 CPU 上跑。
EMA Centroid	GlobalTrackMeta (System RAM)	单个 float[512] 数组	体积极小（2KB），随轨迹对象的创建/销毁同步管理，无需复杂的 Pool 逻辑。
结合你提供的全景拼接图（从图中可以看出，三个摄像头的视野边缘仅有极小范围的重叠区域） 以及你目前的 `association_develop.md` 架构，将 PaddleDetection 的**“级联匹配” (Cascade Matching)** 理念引入跨摄像头（MTMC）场景是一个非常绝妙的升级。


### 核心思想：从“单维度比对”升级为“时空+特征的漏斗过滤” 
参考https://gitee.com/paddlepaddle/PaddleDetection/tree/release/2.8.1/deploy/pptracking/cpp/src实现
我们保留你原有的 `GlobalTrackMeta` 结构，但将匹配流程重构为以下四个级联阶段（Cascade Stages）：

---

#### 阶段 0：单摄内极速关联 (Intra-Camera Local Track)

这是你的第一道防线，完全不依赖复杂的全局计算。

* **逻辑**：系统收到某个 Detection 后，优先按 `(camera_id, local_track_id)` 在 Global Gallery 中寻找处于 `ACTIVE` 状态的轨迹。
* **操作**：如果找到，直接将当前 L2 归一化后的 Embedding 用于更新该轨迹的 EMA Centroid 和环形队列。
* **收益**：过滤掉 80% 以上在单一画面中稳定移动的目标，极大地缩小后续跨摄匹配的计算规模。

---

#### 阶段 1：跨摄时空拓扑门控 (Spatio-Temporal Gating) - 构建代价矩阵的骨架

对于阶段 0 剩下的 Unmatched Detections 和 Global Gallery 中的 Lost Tracks，我们准备构建代价矩阵，但在计算 Re-ID 之前，先上“物理外挂”。

* **逻辑**：根据摄像头的物理安装位置，建立一个**转移概率矩阵 (Transition Graph)**。
* 例如，目标在图左侧画面消失，绝不可能在 1 秒内出现在图右侧画面的右边缘。


* **操作**：初始化一个大小为 $M \times N$ 的代价矩阵。如果 Detection $i$ 和 Track $j$ 在时间和空间拓扑上绝对不可能关联，直接在矩阵中填入 `INF`（无穷大），**直接跳过后续的余弦计算**。

---

#### 阶段 2：高置信度特征匹配 (Cascade 1: High-Confidence Re-ID)

这是借鉴 Paddle 第一级匹配的核心，也是你 `association_develop.md` 逻辑的发力点。

* **逻辑**：对时空门控判定为“可能”的对 $(i, j)$ 进行双层特征打分。
1. **粗筛**：用 `centroid_embedding` 算点积。如果 $< 0.6$，代价矩阵填 `INF`。
2. **精打分**：对通过粗筛的，遍历该 Track 的环形队列，取最大相似度（如 0.85），计算 $Cost = 1.0 - 0.85 = 0.15$ 填入矩阵。


* **求解**：将这个稀疏的代价矩阵送入 **匈牙利算法 (LAPJV/KM)** 进行全局最优分配。
* **严格阈值**：此阶段必须设置一个严格的匹配阈值（例如 Cost < 0.3 即相似度 > 0.7）。只有特征高度确信的目标才在此阶段被关联。

---

#### 阶段 3：边缘交接与盲区兜底 (Cascade 2: Edge & Blind-Spot Fallback)

这是应对你“小重叠区域”最关键的一步。对应 Paddle 中的 IoU 兜底，但在 MTMC 中，我们用**“边界转移区 (Hand-over Zone)”**兜底。

* **痛点**：目标在画面接缝处（图中的红线附近）因截断或畸变，导致 Re-ID 相似度只有 0.4，在阶段 2 中被抛弃了。
* **逻辑**：取出阶段 2 剩下的 Unmatched Detections 和 Unmatched Tracks。此时我们**放宽特征要求**，极度依赖空间位置。
* 如果 Track $j$ 是在 Camera A 的**右边界**丢失的。
* Detection $i$ 刚好在极短时间（如 2 秒）内出现在 Camera B 的**左边界**。


* **操作**：重新构建一个小型的代价矩阵，特征相似度权重降低，时空转移权重拉满。再次运行一次**匈牙利算法**。这能把那些因为“走在缝隙里变形”而断掉的轨迹强行接上。

---

#### 阶段 4：轨迹生命周期管理 (Track Lifecycle)

效仿 Paddle 的状态机，你需要为 `GlobalTrackMeta` 引入明确的状态：

1. **未匹配到的 Detections**：实例化为新的 `GlobalTrackMeta`，状态设为 `TENTATIVE`（观察期），连续存在 3 帧才转为 `ACTIVE`（避免误检噪点污染全局库）。
2. **未匹配到的 Tracks**：状态从 `ACTIVE` 切为 `LOST`。保留其 `centroid_embedding` 和环形队列在 System RAM 中。如果 `LOST` 状态超过阈值（如 300 帧 / 10 秒），则彻底销毁（Memory Free），防止系统内存溢出。

### 总结：你的系统获得了什么提升？

通过这种级联设计：

1. **极低的 CPU 负载**：绝大多数目标在阶段 0 被过滤；跨摄目标在阶段 1 被拓扑门控拦截；真正进入阶段 2 遍历环形队列并跑匈牙利算法的矩阵极其稀疏（可能只是个 $3 \times 5$ 的小矩阵），在 Orin 的 CPU 上耗时几乎可以忽略不计。
2. **抗畸变能力**：阶段 3 的“边缘交接兜底”完美弥补了你这种小重叠多摄场景下，目标穿过红线接缝时产生的特征退化问题。

**你需要我为你提供一个定义了 `ACTIVE/LOST` 状态机，并且包含两级代价矩阵构建逻辑的 C++ 头文件 (Header) 结构草案吗？**