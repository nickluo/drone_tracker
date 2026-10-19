#include "hybrid_tracker.h"
#include <iostream>
#include <deque>
#include <vector>
#include <opencv2/opencv.hpp>
#include <vpi/VPI.h>

#include "yolo_detector.h"
#include "nano_tracker.h"
#include "dcf_tracker.h"
#include "rotation_kf.h"

/**
 * @brief airsim_tracker_node.process_frame_callback 的移植。
 *
 * 每帧流程：时间戳单调性检查 -> KF 预测(只读) -> 补帧缓存 ->
 * 检测/跟踪分支（跳变抑制 + YOLO 再检测仲裁）-> KF 校正/重置 ->
 * predict_only(now) 实时外推 -> 返回结果。
 */
class HybridTrackerImp : public HybridTracker
{
public:
    HybridTrackerImp(const TrackerSettings& settings);
    virtual ~HybridTrackerImp();

    virtual bool init() override;
    virtual void reset() override;
    virtual TrackerOutput update(const FrameContext& ctx) override;
    virtual cv::Rect2f lastNanoBox() const override { return m_tracker ? m_tracker->lastNanoBox() : cv::Rect2f(); }
    virtual float lastNanoScore() const override { return m_tracker ? m_tracker->lastScore() : -1.0f; }

private:
    /** @brief 补帧缓存条目：帧号 + 帧 + 当时的跟踪框/KF 预测框 */
    struct BufferedFrame
    {
        uint64_t frame_id = 0;
        std::shared_ptr<VPIImageWrapper> frame;
        cv::Rect2f tracker_box;
        bool has_tracker_box = false;
        cv::Rect2f kf_pred_box;
        bool has_kf_pred = false;
        int64_t ts_us = 0;
    };

    /** @brief 用历史帧初始化跟踪器并对后续帧前向追踪以追平进度 */
    bool catchupTracker(const BufferedFrame& hist,
                        const std::vector<std::shared_ptr<VPIImageWrapper>>& catchup_frames,
                        const cv::Rect2f& yolo_box);

    /** @brief 计算两个矩形的 IoU */
    static float calculateIOU(const cv::Rect2f& boxA, const cv::Rect2f& boxB);

    /** @brief 检测边界框是否发生跳变（中心距离超阈值 或 尺寸比例超界） */
    static bool isBoxJump(const cv::Rect2f& prev_box, const cv::Rect2f& curr_box,
                          float max_pos_jump, float max_size_ratio);

    /** @brief 计算 ROI 区域（Python calculate_roi 语义：int 截断 + min_size 下限 + 裁剪） */
    static cv::Rect calculateROI(const cv::Rect2f& box, int img_w, int img_h,
                                 float scale_factor, const cv::Size& min_size);

    static cv::Rect2f toIntRect(const cv::Rect2f& r)
    {
        return cv::Rect2f(static_cast<int>(r.x), static_cast<int>(r.y),
                          static_cast<int>(r.width), static_cast<int>(r.height));
    }

    static BBoxRect toBBoxRect(const cv::Rect2f& r, bool valid)
    {
        return BBoxRect{
            static_cast<int>(r.x + r.width * 0.5f + 0.5f),
            static_cast<int>(r.y + r.height * 0.5f + 0.5f),
            static_cast<int>(r.width + 0.5f),
            static_cast<int>(r.height + 0.5f),
            valid && r.width > 0 && r.height > 0};
    }

    void clearBoxes()
    {
        m_has_display_box = false;
        m_has_previous_box = false;
        m_has_raw_tracker_box = false;
    }

    // --- 设置 ---
    TrackerSettings m_settings;

    // --- 状态 ---
    bool m_tracker_initialized = false;
    cv::Rect2f m_display_box;          // 整数化显示框（Python: tuple(map(int, ...))）
    bool m_has_display_box = false;
    cv::Rect2f m_previous_box;
    bool m_has_previous_box = false;
    cv::Rect2f m_raw_tracker_box;      // 原始 float 跟踪结果，喂 KF 避免整数量化噪声
    bool m_has_raw_tracker_box = false;
    std::string m_display_info = "Detecting...";
    cv::Scalar m_display_color = cv::Scalar(0, 255, 255); // 黄色 (BGR)
    int m_frame_count = 0;
    uint64_t m_frame_id_counter = 0;
    int64_t m_last_frame_ts_us = 0;
    int m_jump_counter = 0;            // 跟踪框连续跳变计数
    int m_yolo_jump_count = 0;         // YOLO 检测连续跳变计数
    int m_yolo_lost = 0;               // YOLO 连续未确认次数（无目标/低IoU重初始化/未确认跳变）
    int m_init_fail_streak = 0;         // 检测初始化连续失败次数（防死循环保护）
    int m_no_detection_frames = 0;     // 未收到任何检测信号的帧数（TRT 卡死保护）
    TrackerState m_state = TrackerState::Detecting;

    std::deque<BufferedFrame> m_frame_buffer;

    // --- 组件 ---
    std::unique_ptr<YoloV8Detector> m_detector;
    std::unique_ptr<NanoDcfTracker> m_tracker;
    std::unique_ptr<RotationKalmanFilter> m_kf;
};

std::shared_ptr<HybridTracker> HybridTracker::CreateInstance(const TrackerSettings& settings)
{
    return std::make_shared<HybridTrackerImp>(settings);
}

HybridTrackerImp::HybridTrackerImp(const TrackerSettings& settings)
    : m_settings(settings)
{
}

HybridTrackerImp::~HybridTrackerImp()
{
    // m_detector 将在其析构函数中自动停止其线程
}

bool HybridTrackerImp::init()
{
    m_detector = std::make_unique<YoloV8Detector>(
        m_settings.yolo_engine_path,
        m_settings.yolo_conf_threshold,
        m_settings.yolo_nms_threshold,
        m_settings.target_class_id,
        m_settings.yolo_recall_relax_ratio,
        m_settings.max_box_area_ratio
    );

    if (!m_detector || !m_detector->init())
    {
        std::cerr << "Error: Failed to initialize YoloV8Detector." << std::endl;
        return false;
    }

    NanoDcfTracker::Params tracker_params;
    tracker_params.backbone_path = m_settings.nanotrack_backbone_path;
    tracker_params.head_path = m_settings.nanotrack_head_path;
    tracker_params.enable_dcf = m_settings.enable_dcf;
    tracker_params.dcf = m_settings.dcf;
    tracker_params.max_div_factor = m_settings.dcf_max_div_factor;
    // enable_dcf=false 时的中心平滑 KF 噪声（与主管线 KF 共用调参入口）
    tracker_params.center_kf.r_pos = m_settings.kf_r_pos;
    tracker_params.center_kf.q_a = m_settings.kf_q_a;
    m_tracker = std::make_unique<NanoDcfTracker>(tracker_params);

    reset();
    return true;
}

void HybridTrackerImp::reset()
{
    if (!m_detector || !m_tracker)
        return;

    RotationKalmanFilter::Params kf_params;
    kf_params.fx = m_settings.kf_fx;
    kf_params.fy = m_settings.kf_fy;
    kf_params.cx0 = m_settings.kf_cx0;
    kf_params.cy0 = m_settings.kf_cy0;
    kf_params.r_pos = m_settings.kf_r_pos;
    kf_params.r_size = m_settings.kf_r_size;
    kf_params.q_a = m_settings.kf_q_a;
    kf_params.q_size = m_settings.kf_q_size;
    kf_params.history_len = m_settings.kf_history_len;
    m_kf = std::make_unique<RotationKalmanFilter>(kf_params);

    // 只清空跟踪状态，TRT 引擎保留（Python can_reuse_nanotrack 的等价物）
    m_tracker->reset();

    m_tracker_initialized = false;
    clearBoxes();
    m_frame_count = 0;
    m_frame_id_counter = 0;
    m_last_frame_ts_us = 0;
    m_frame_buffer.clear();
    m_jump_counter = 0;
    m_yolo_jump_count = 0;
    m_yolo_lost = 0;
    m_init_fail_streak = 0;
    m_no_detection_frames = 0;
    m_state = TrackerState::Detecting;
    m_display_info = "Detecting...";
    m_display_color = cv::Scalar(0, 255, 255); // 黄色 (BGR)
}

bool HybridTrackerImp::catchupTracker(const BufferedFrame& hist,
                                       const std::vector<std::shared_ptr<VPIImageWrapper>>& catchup_frames,
                                       const cv::Rect2f& yolo_box)
{
    if (!m_tracker->init(hist.frame, yolo_box))
    {
        m_tracker->reset();
        return false;
    }
    m_display_box = toIntRect(yolo_box);
    m_has_display_box = true;
    m_previous_box = m_display_box;
    m_has_previous_box = true;

    for (const auto& f : catchup_frames)
    {
        cv::Rect2f t_box;
        if (m_tracker->update(f, t_box))
        {
            m_display_box = toIntRect(t_box);
            m_previous_box = m_display_box;
        }
    }
    return true;
}

TrackerOutput HybridTrackerImp::update(const FrameContext& ctx)
{
    TrackerOutput out;
    out.state = m_state;
    out.info = m_display_info;
    out.color = (uint32_t(m_display_color[2]) << 16) | (uint32_t(m_display_color[1]) << 8) | uint32_t(m_display_color[0]);

    if (!m_detector || !m_tracker || !m_kf || !ctx.frame)
        return out;

    // 防止重复帧或时间倒退导致卡尔曼滤波协方差坍缩
    if (ctx.frame_ts_us <= m_last_frame_ts_us)
        return out; // consumed = false，静默丢弃
    m_last_frame_ts_us = ctx.frame_ts_us;

    m_frame_count++;
    m_frame_id_counter++;
    const uint64_t current_frame_id = m_frame_id_counter;

    int frame_w = 0, frame_h = 0;
    vpiImageGetSize(ctx.frame->get(), &frame_w, &frame_h);

    // 提前获取卡尔曼滤波的预测框，用于本帧的跳变检测和 ROI 生成。
    // 注意：KF 预测是 (cx, cy, w, h) 中心格式，而下游 calculateROI/isBoxJump
    // 按左上角格式解析——与 Python 参考实现保持逐行为一致（其语义如此）。
    cv::Vec4d kf_pred;
    const bool has_kf_pred = m_kf->predictOnly(ctx.frame_ts_us, ctx.R_cam_at_frame, kf_pred);
    cv::Rect2f kf_pred_box;
    if (has_kf_pred)
        kf_pred_box = cv::Rect2f(kf_pred[0], kf_pred[1], kf_pred[2], kf_pred[3]);

    // 维护缓存队列（保存最新 frame_buffer_size 帧）
    BufferedFrame buffered;
    buffered.frame_id = current_frame_id;
    buffered.frame = ctx.frame;
    buffered.tracker_box = m_display_box;
    buffered.has_tracker_box = m_has_display_box;
    buffered.kf_pred_box = kf_pred_box;
    buffered.has_kf_pred = has_kf_pred;
    buffered.ts_us = ctx.frame_ts_us;
    m_frame_buffer.push_back(buffered);
    if (static_cast<int>(m_frame_buffer.size()) > m_settings.frame_buffer_size)
        m_frame_buffer.pop_front();

    bool measurement_updated = false;
    bool kf_needs_reset = false;

    // ------------------------------------------------------------------
    // 分支 1：跟踪器未初始化 -> 检测（ROI(KF预测) -> 全图 -> 切片 由 worker 完成）
    // ------------------------------------------------------------------
    if (!m_tracker_initialized)
    {
        m_display_info = "Detecting...";
        m_display_color = cv::Scalar(0, 255, 255); // 黄色
        m_state = TrackerState::Detecting;

        if (!m_detector->isBusy())
        {
            cv::Rect roi;
            if (has_kf_pred)
            {
                // 即使追踪器未初始化，如果有 KF 预测，也可以用作 ROI！
                const auto [in_w, in_h] = m_detector->getInputSize();
                roi = calculateROI(kf_pred_box, frame_w, frame_h,
                                   m_settings.roi_scale_factor * 1.5f, cv::Size(in_w / 2, in_h / 2));
            }
            m_detector->request(ctx.frame, current_frame_id, roi);
        }

        YoloResult res;
        if (m_detector->get_result(res) && res.has_target)
        {
            // 找到检测结果对应的缓存帧，补帧初始化并前向追平
            bool found = false;
            size_t idx = 0;
            for (size_t i = 0; i < m_frame_buffer.size(); ++i)
            {
                if (m_frame_buffer[i].frame_id == res.frame_id)
                {
                    found = true;
                    idx = i;
                    break;
                }
            }

            if (found)
            {
                std::vector<std::shared_ptr<VPIImageWrapper>> catchup_frames;
                for (size_t i = idx + 1; i < m_frame_buffer.size(); ++i)
                    catchup_frames.push_back(m_frame_buffer[i].frame);

                if (catchupTracker(m_frame_buffer[idx], catchup_frames, res.best.box))
                {
                    m_yolo_lost = 0;
                    m_init_fail_streak = 0;
                    m_tracker_initialized = true;
                    kf_needs_reset = true;
                    m_state = TrackerState::TargetFound;
                    m_display_info = "Target Found";
                    m_display_color = cv::Scalar(0, 255, 255);
                    std::cout << "Tracker initialized at [" << res.best.box.x << ", " << res.best.box.y
                              << ", " << res.best.box.width << ", " << res.best.box.height
                              << "] and caught up " << catchup_frames.size() << " frames" << std::endl;
                }
                else
                {
                    // 初始化失败保护：连续失败多半因为 KF 预测漂移导致 ROI
                    // 停在错误区域——重置 KF，让检测回到全图/切片级联
                    if (++m_init_fail_streak >= 10)
                    {
                        std::cout << "Tracker init failed " << m_init_fail_streak
                                  << " times in a row, resetting KF and detection ROI" << std::endl;
                        m_kf->reset();
                        m_init_fail_streak = 0;
                    }
                }
            }
        }
    }
    // ------------------------------------------------------------------
    // 分支 2：已初始化 -> 跟踪 + 定期 YOLO 再检测仲裁
    // ------------------------------------------------------------------
    else
    {
        bool ok = false;
        cv::Rect2f tracker_box;
        ok = m_tracker->update(ctx.frame, tracker_box);

        if (ok)
        {
            const cv::Rect2f tracker_box_i = toIntRect(tracker_box);

            // 跳变检测：跟踪框需同时偏离 KF 预测位置与上一帧位置才判定失效
            bool jump_from_kf = true;
            if (has_kf_pred)
                jump_from_kf = isBoxJump(kf_pred_box, tracker_box_i,
                                         m_settings.max_position_jump, m_settings.max_size_jump_ratio);

            bool jump_from_prev = true;
            if (m_has_previous_box)
                jump_from_prev = isBoxJump(m_previous_box, tracker_box_i,
                                           m_settings.max_position_jump, m_settings.max_size_jump_ratio);

            if (jump_from_kf && jump_from_prev)
            {
                m_jump_counter++;
                std::cout << "Box jump detected! curr=[" << tracker_box_i.x << ", " << tracker_box_i.y
                          << ", " << tracker_box_i.width << ", " << tracker_box_i.height
                          << "] (counter=" << m_jump_counter << ")" << std::endl;

                // 连续多次跳变才重置，否则本帧不喂 KF、保持旧显示框
                if (m_jump_counter >= m_settings.jump_tolerance)
                {
                    std::cout << "Persistent jump, re-initializing tracker..." << std::endl;
                    m_tracker_initialized = false;
                    clearBoxes();
                    m_jump_counter = 0;
                    // 提前返回：本帧不发布（Python 的 return True 早退）
                    return out;
                }
                m_display_info = "KF Predict (Jump)";
                m_display_color = cv::Scalar(0, 165, 255); // 橙色
                // measurement_updated 保持 false
            }
            else
            {
                m_jump_counter = 0;
                m_display_box = tracker_box_i;
                m_has_display_box = true;
                m_previous_box = tracker_box_i;
                m_has_previous_box = true;
                // 保存原始 float 的追踪结果，避免整数截断引入量化噪声
                m_raw_tracker_box = tracker_box;
                m_has_raw_tracker_box = true;
                measurement_updated = true;
                m_display_info = "Tracking (nanotrack)";
                m_display_color = cv::Scalar(0, 255, 0); // 绿色
            }
            m_state = TrackerState::Tracking;
        }
        else
        {
            // 追踪丢失，不喂给 KF
            m_display_info = "Predicting";
            m_display_color = cv::Scalar(0, 165, 255); // 橙色
            m_state = TrackerState::Tracking;
        }

        // 定期请求 YOLO 检测（ROI 优先）
        if (m_frame_count % m_settings.yolo_interval == 0)
        {
            if (!m_detector->isBusy())
            {
                cv::Rect roi;
                const cv::Rect2f roi_base = has_kf_pred ? kf_pred_box
                                           : (m_has_display_box ? m_display_box : cv::Rect2f());
                if (roi_base.width > 0 && roi_base.height > 0)
                {
                    const auto [in_w, in_h] = m_detector->getInputSize();
                    roi = calculateROI(roi_base, frame_w, frame_h,
                                       m_settings.roi_scale_factor, cv::Size(in_w / 2, in_h / 2));
                }
                m_detector->request(ctx.frame, current_frame_id, roi);
            }
        }

        // 检查 YOLO 检测结果
        YoloResult res;
        if (m_detector->get_result(res))
        {
            m_no_detection_frames = 0; // 收到检测信号，重置无信号计数
            if (res.has_target)
            {
                const cv::Rect2f& best_match_box = res.best.box;

                // 获取历史帧对应的 tracker_box / kf_pred_box
                bool found = false;
                size_t idx = 0;
                for (size_t i = 0; i < m_frame_buffer.size(); ++i)
                {
                    if (m_frame_buffer[i].frame_id == res.frame_id)
                    {
                        found = true;
                        idx = i;
                        break;
                    }
                }
                const BufferedFrame* hist = found ? &m_frame_buffer[idx] : nullptr;
                std::vector<std::shared_ptr<VPIImageWrapper>> catchup_frames;
                if (found)
                {
                    for (size_t i = idx + 1; i < m_frame_buffer.size(); ++i)
                        catchup_frames.push_back(m_frame_buffer[i].frame);
                }

                // 边缘目标仲裁：目标贴边/半出画时，YOLO 只能看到可见部分而
                // 跟踪器估计完整外接框，两者对"不可见部分"的估计天然不同——
                // 直接比较会导致 IoU 偏低、跳变误判，进而反复重初始化/判丢，
                // 表现为边缘 bbox 抖动。这里把参与比较的框统一裁剪到图像范围，
                // 只比较可见部分。
                const cv::Rect2f image_rect(0.0f, 0.0f,
                                            static_cast<float>(frame_w), static_cast<float>(frame_h));
                auto clipped = [&image_rect](cv::Rect2f r) { r &= image_rect; return r; };
                const cv::Rect2f yolo_box_vis = clipped(best_match_box);
                const cv::Rect2f hist_tracker_vis =
                    clipped(hist && hist->has_tracker_box ? hist->tracker_box
                               : (m_has_display_box ? m_display_box : cv::Rect2f()));

                // 用历史框计算 IOU（可见部分），没找到历史帧则回退用当前框
                float current_iou = 0.0f;
                if ((hist && hist->has_tracker_box) || m_has_display_box)
                    current_iou = calculateIOU(hist_tracker_vis, yolo_box_vis);

                // YOLO 检测结果跳变检测（可能检测到了其他物体）：
                // 同时偏离历史 KF 预测与历史跟踪框才算跳变（均按可见部分比较）
                bool yolo_jump_kf = true;
                if (hist && hist->has_kf_pred)
                    yolo_jump_kf = isBoxJump(clipped(hist->kf_pred_box), yolo_box_vis,
                                             m_settings.max_position_jump, m_settings.max_size_jump_ratio);
                bool yolo_jump_prev = true;
                if (hist && hist->has_tracker_box)
                    yolo_jump_prev = isBoxJump(hist_tracker_vis, yolo_box_vis,
                                               m_settings.max_position_jump, m_settings.max_size_jump_ratio);
                else if (m_has_previous_box)
                    yolo_jump_prev = isBoxJump(clipped(m_previous_box), yolo_box_vis,
                                               m_settings.max_position_jump, m_settings.max_size_jump_ratio);
                const bool yolo_jump = yolo_jump_kf && yolo_jump_prev;

                if (yolo_jump && current_iou < m_settings.iou_calibrate_threshold)
                {
                    m_yolo_jump_count++;
                    std::cout << "YOLO detection jump (" << m_yolo_jump_count << ")! IOU="
                              << current_iou << std::endl;
                    // 连续多次"跳变"说明目标确实移动了，应该接受新检测
                    if (m_yolo_jump_count >= m_settings.jump_tolerance)
                    {
                        std::cout << "Accepting jumped YOLO detection after consecutive jumps" << std::endl;
                        if (hist && catchupTracker(*hist, catchup_frames, best_match_box))
                        {
                            m_display_info = "Re-initialized (jump accept)";
                            m_display_color = cv::Scalar(255, 255, 0); // 青色
                            kf_needs_reset = true;
                        }
                        m_yolo_jump_count = 0;
                        m_yolo_lost = 0;   // 接受跳变 = 主动跟随移动目标，重置未确认计数
                    }
                    else
                    {
                        m_yolo_lost++;     // 未获确认的跳变检测
                    }
                }
                else if (!ok || current_iou < m_settings.iou_reset_threshold)
                {
                    m_yolo_jump_count = 0;
                    // 重新初始化追踪器（复用已加载的 TRT 引擎）
                    if (hist)
                    {
                        if (catchupTracker(*hist, catchup_frames, best_match_box))
                        {
                            m_display_info = "Re-initialized (nanotrack)";
                            m_display_color = cv::Scalar(255, 255, 0); // 青色
                            kf_needs_reset = true;
                        }
                        else
                        {
                            std::cerr << "Failed to re-initialize tracker" << std::endl;
                        }
                    }
                    m_yolo_lost++;         // 低 IoU 重初始化：本帧未获确认
                }
                // IOU 足够高时不覆盖当前追踪框，避免引入滞后
                else if (current_iou > m_settings.iou_calibrate_threshold)
                {
                    m_yolo_jump_count = 0;
                    m_yolo_lost = 0;       // YOLO 确认
                    m_display_info = "YOLO Confirmed";
                    m_display_color = cv::Scalar(0, 0, 255); // 红色
                }
                else
                {
                    m_yolo_jump_count = 0;
                    m_yolo_lost = 0;       // 中间带：跟踪器与检测保持合理重叠
                }
            }
            else
            {
                // YOLO 没有检测到目标
                m_yolo_jump_count = 0;
                m_yolo_lost++;
            }

            // 连续未确认（无目标 / 低 IoU 重初始化 / 未确认跳变）达到上限 ->
            // 判定丢失。仅有"无目标"判据不足以覆盖 YOLO 持续误检的场景：
            // 目标实际丢失后，误检会让跟踪器在垃圾框上无限重初始化而永不报丢。
            if (m_tracker_initialized && m_yolo_lost >= m_settings.lost_max_count)
            {
                m_tracker_initialized = false;
                clearBoxes();
                std::cout << "Target lost (unconfirmed streak " << m_yolo_lost << ")."
                          << " Re-initializing detection." << std::endl;
                m_state = TrackerState::TargetLost;
            }
        }
        else
        {
            // 没有收到检测信号，累加无信号计数；长时间无任何信号（TRT 线程
            // 可能卡死）则强制重置
            m_no_detection_frames++;
            const int no_detect_limit = m_settings.lost_max_count * m_settings.yolo_interval * 3;
            if (m_no_detection_frames > no_detect_limit)
            {
                std::cout << "No YOLO detection signal for " << m_no_detection_frames
                          << " frames, forcing tracker reset..." << std::endl;
                m_tracker_initialized = false;
                clearBoxes();
                m_no_detection_frames = 0;
            }
        }
    }

    // 更新当前帧在缓存中的 tracker_box，以备未来 YOLO 参考
    if (!m_frame_buffer.empty())
    {
        m_frame_buffer.back().tracker_box = m_display_box;
        m_frame_buffer.back().has_tracker_box = m_has_display_box;
    }

    // ------------------------------------------------------------------
    // KF 校正/重置 + 实时外推
    // ------------------------------------------------------------------
    if (m_has_display_box)
    {
        if (kf_needs_reset)
        {
            // 跟踪器刚被 YOLO 拉回：重置 KF 以清空错误的旧速度
            const float rx = m_display_box.x, ry = m_display_box.y;
            const float rw = m_display_box.width, rh = m_display_box.height;
            m_kf->initialize(cv::Vec4d(rx + rw / 2.0, ry + rh / 2.0, rw, rh),
                             ctx.frame_ts_us, ctx.R_cam_at_frame);
            out.reset_los = true;
        }
        else if (measurement_updated && m_has_raw_tracker_box)
        {
            // 使用原始 float 追踪结果校正（回溯到帧时刻）
            const float rx = m_raw_tracker_box.x, ry = m_raw_tracker_box.y;
            const float rw = m_raw_tracker_box.width, rh = m_raw_tracker_box.height;
            m_kf->update(cv::Vec4d(rx + rw / 2.0, ry + rh / 2.0, rw, rh),
                         ctx.frame_ts_us, ctx.R_cam_at_frame);
        }

        // predict_only 查询实时位置（不修改滤波器状态和历史）。
        // 发布框裁剪到图像范围：贴边目标的外推可能越界，下游不应收到
        // 图像外的测量（内部 KF 状态保持不裁剪，目标回到画内可自然恢复）
        cv::Vec4d realtime;
        if (m_kf->predictOnly(ctx.now_us, ctx.R_cam_now, realtime))
        {
            cv::Rect2d rt_box(realtime[0] - realtime[2] / 2.0,
                              realtime[1] - realtime[3] / 2.0,
                              realtime[2], realtime[3]);
            rt_box &= cv::Rect2d(0.0, 0.0, frame_w, frame_h);
            out.observation_box = rt_box;
            out.realtime_box = toBBoxRect(cv::Rect2f(rt_box), rt_box.area() > 0);
        }
    }

    out.consumed = true;
    out.tracking = m_tracker_initialized && m_has_display_box;
    {
        cv::Rect2f display = m_display_box;
        display &= cv::Rect2f(0.0f, 0.0f,
                              static_cast<float>(frame_w), static_cast<float>(frame_h));
        out.display_box = toBBoxRect(display, m_has_display_box);
    }
    out.state = m_state;
    out.info = m_display_info;
    out.color = (uint32_t(m_display_color[2]) << 16) | (uint32_t(m_display_color[1]) << 8) | uint32_t(m_display_color[0]);
    return out;
}

float HybridTrackerImp::calculateIOU(const cv::Rect2f& boxA, const cv::Rect2f& boxB)
{
    const float xA = std::max(boxA.x, boxB.x);
    const float yA = std::max(boxA.y, boxB.y);
    const float xB = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    const float yB = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    const float interArea = std::max(0.0f, xB - xA) * std::max(0.0f, yB - yA);
    const float boxAArea = boxA.width * boxA.height;
    const float boxBArea = boxB.width * boxB.height;

    const float denominator = boxAArea + boxBArea - interArea;
    return (denominator == 0) ? 0.0f : (interArea / denominator);
}

bool HybridTrackerImp::isBoxJump(const cv::Rect2f& prev_box, const cv::Rect2f& curr_box,
                                 float max_pos_jump, float max_size_ratio)
{
    if (prev_box.width <= 0 || prev_box.height <= 0 || curr_box.width <= 0 || curr_box.height <= 0)
        return false;

    const float prev_cx = prev_box.x + prev_box.width / 2.0f;
    const float prev_cy = prev_box.y + prev_box.height / 2.0f;
    const float curr_cx = curr_box.x + curr_box.width / 2.0f;
    const float curr_cy = curr_box.y + curr_box.height / 2.0f;

    const float distance = std::sqrt(std::pow(curr_cx - prev_cx, 2) + std::pow(curr_cy - prev_cy, 2));

    const float size_ratio_w = curr_box.width / prev_box.width;
    const float size_ratio_h = curr_box.height / prev_box.height;

    const bool pos_jump = distance > max_pos_jump;
    const bool size_jump = (size_ratio_w > max_size_ratio || size_ratio_w < 1.0f / max_size_ratio ||
                            size_ratio_h > max_size_ratio || size_ratio_h < 1.0f / max_size_ratio);

    return pos_jump || size_jump;
}

cv::Rect HybridTrackerImp::calculateROI(const cv::Rect2f& box, int img_w, int img_h,
                                        float scale_factor, const cv::Size& min_size)
{
    const float cx = box.x + box.width * 0.5f;
    const float cy = box.y + box.height * 0.5f;

    int roi_w = static_cast<int>(box.width * scale_factor);
    int roi_h = static_cast<int>(box.height * scale_factor);

    // 确保ROI不小于最小尺寸
    if (min_size.width > 0)
        roi_w = std::max(roi_w, min_size.width);
    if (min_size.height > 0)
        roi_h = std::max(roi_h, min_size.height);

    int roi_x = static_cast<int>(cx - roi_w / 2.0f);
    int roi_y = static_cast<int>(cy - roi_h / 2.0f);

    // 确保ROI在图像范围内
    roi_x = std::max(0, std::min(roi_x, img_w - roi_w));
    roi_y = std::max(0, std::min(roi_y, img_h - roi_h));
    roi_w = std::min(roi_w, img_w - roi_x);
    roi_h = std::min(roi_h, img_h - roi_y);

    return cv::Rect(roi_x, roi_y, roi_w, roi_h);
}
