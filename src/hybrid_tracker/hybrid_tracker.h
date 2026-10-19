#ifndef HYBRIDTRACKER_H
#define HYBRIDTRACKER_H

#include <string>
#include <memory>
#include <cstdint>
#include <opencv2/core.hpp>

#include "dcf_tracker.h"

// 用于配置 HybridTracker 的结构体（默认值对齐 airsim_tracker launch.py）
struct TrackerSettings
{
    std::string yolo_engine_path = "drone_int8.engine";
    std::string nanotrack_backbone_path = "nanotrackv3_backbone_int8.engine";
    std::string nanotrack_head_path = "nanotrack_head_int8.engine";
    int target_class_id = 0;                 // 目标类别：接受 cid >= target_class_id
    float yolo_conf_threshold = 0.4f;
    float yolo_nms_threshold = 0.4f;
    float yolo_recall_relax_ratio = 1.0f;    // ROI 阶段置信度放宽比例（1.0 = 关闭，Python 无此机制）
    int yolo_interval = 4;                   // 每 N 帧运行一次 YOLO 再检测
    float iou_reset_threshold = 0.4f;        // IoU 低于此值时重置跟踪器
    float iou_calibrate_threshold = 0.8f;    // IoU 高于此值时视为 YOLO 确认（不干预跟踪）
    float max_position_jump = 150.0f;        // 最大位置跳变（像素，绝对值）
    float max_size_jump_ratio = 2.0f;        // 最大尺寸跳变比例
    float max_box_area_ratio = 0.8f;         // 检测框边长/图像边长上限
    int lost_max_count = 5;                  // YOLO 连续未检测到目标的次数上限
    float roi_scale_factor = 2.0f;
    int frame_buffer_size = 30;              // 补帧缓存长度
    int jump_tolerance = 3;                  // 连续跳变多少次后强制重置
    bool enable_dcf = true;                  // DCF 融合开关（A/B 对比用）
    GrayscaleDCF::Params dcf;                // DCF 参数
    float dcf_max_div_factor = 0.5f;
    // KF 内参与噪声（节点从 mv_camera.* 推导焦距/主点后填充）
    double kf_fx = 0.0;
    double kf_fy = 0.0;
    double kf_cx0 = 0.0;
    double kf_cy0 = 0.0;
    double kf_r_pos = 4.0;
    double kf_r_size = 4.0;
    double kf_q_a = 5e4;
    double kf_q_size = 1e3;
    int kf_history_len = 60;
};

class VPIImageWrapper;

struct BBoxRect
{
    int cx; //!< x coordinate of the center
    int cy; //!< y coordinate of the center
    int width; //!< width of the rectangle
    int height; //!< height of the rectangle
    bool is_valid;
};

enum class TrackerState : int8_t
{
    Detecting = 0,
    TargetFound,
    Tracking,
    TargetLost
};

/** @brief 单帧输入上下文 */
struct FrameContext
{
    std::shared_ptr<VPIImageWrapper> frame;
    int64_t frame_ts_us = 0;    // 帧采集时刻（系统时钟，us）
    int64_t now_us = 0;         // 发布参考时刻：采集时刻或最新的新鲜云台姿态时刻
    cv::Matx33d R_cam_at_frame = cv::Matx33d::eye(); // 帧时刻相机旋转（无云台数据时为单位阵）
    cv::Matx33d R_cam_now = cv::Matx33d::eye();      // 当前相机旋转
};

/** @brief 单帧处理结果 */
struct TrackerOutput
{
    bool consumed = false;      // false: 重复帧/时间倒退/持续跳变提前返回（跳过本帧发布）
    bool tracking = false;
    BBoxRect display_box{};     // 整数显示框（跟踪器结果，用于可视化）
    BBoxRect realtime_box{};    // KF 在发布参考时刻的整数显示框
    cv::Rect2d observation_box{}; // 同一预测的浮点框，保留亚像素精度用于 bbox/LOS
    bool reset_los = false;     // YOLO 重锚定清空旧 LOS 角速度
    TrackerState state = TrackerState::Detecting;
    std::string info;           // 状态文本（可视化/StatusText）
    uint32_t color = 0;         // 显示颜色 RGB 打包
};

/**
 * @class HybridTracker
 * @brief airsim_tracker_node.process_frame_callback 的 C++ 移植：
 * YOLO(TRT 异步) + NanoTrack/DCF 融合 + 旋转补偿回溯卡尔曼滤波，
 * 带补帧缓存、双参考跳变抑制与 YOLO 再检测仲裁。
 */
class HybridTracker
{
public:
    static std::shared_ptr<HybridTracker> CreateInstance(const TrackerSettings& settings);
    virtual ~HybridTracker() = default;

    /** @brief 初始化检测器线程和跟踪器 */
    virtual bool init() = 0;

    /** @brief 触发器 ON 时复位全部状态（TRT 引擎保留） */
    virtual void reset() = 0;

    /** @brief 处理单个视频帧 */
    virtual TrackerOutput update(const FrameContext& ctx) = 0;

    // --- 诊断接口（离线回放/调参用，正常管线不依赖） ---
    /** @brief 本帧 NanoTrack 原始输出框（融合/平滑前）；无有效结果时 width<=0 */
    virtual cv::Rect2f lastNanoBox() const { return {}; }
    /** @brief 本帧 NanoTrack 最优锚点置信度；<=0 表示本帧无推断 */
    virtual float lastNanoScore() const { return -1.0f; }
};


#endif // HYBRIDTRACKER_H
