#ifndef ROTATION_HISTORY_H
#define ROTATION_HISTORY_H

#include <deque>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <opencv2/core.hpp>

/** @brief 单条相机旋转样本：(时间戳 us, 相机旋转矩阵) */
struct RotationSample
{
    int64_t ts_us;
    cv::Matx33d R;
};

/**
 * @brief (ts, R_cam) 滑窗历史，对应 Python 端 odom_history（deque maxlen=100）。
 *
 * 滤波只使用相对旋转 R_new^T * R_old，因此世界系约定任意，但相机系约定必须
 * 前后一致（本工程使用 /fpv/gimbal 的相机到世界、FLU 约定）。
 * 非线程安全——调用方自行加锁。
 */
class RotationHistory
{
public:
    explicit RotationHistory(size_t max_len = 100) : max_len_(max_len) {}

    /** @brief 去重；时钟回退时清空旧 epoch，与 AirSim gimbal_callback 一致 */
    void push(int64_t ts_us, const cv::Matx33d& R)
    {
        if (!samples_.empty() && ts_us == samples_.back().ts_us)
            return;
        if (!samples_.empty() && ts_us < samples_.back().ts_us)
            samples_.clear();
        samples_.push_back({ts_us, R});
        while (samples_.size() > max_len_)
            samples_.pop_front();
    }

    /** @brief 最近邻查找（Python 端 argmin |frame_ts - ts| 的线性扫描） */
    bool nearest(int64_t ts_us, cv::Matx33d& out) const
    {
        if (samples_.empty())
            return false;
        size_t best = 0;
        int64_t best_diff = std::numeric_limits<int64_t>::max();
        for (size_t i = 0; i < samples_.size(); ++i)
        {
            const int64_t diff = std::llabs(samples_[i].ts_us - ts_us);
            if (diff < best_diff)
            {
                best_diff = diff;
                best = i;
            }
        }
        out = samples_[best].R;
        return true;
    }

    bool latest(cv::Matx33d& out) const
    {
        if (samples_.empty())
            return false;
        out = samples_.back().R;
        return true;
    }

    RotationSample predictionReference(int64_t frame_ts_us, const cv::Matx33d& frame_rotation,
                                       int64_t now_us, bool publish_prediction) const
    {
        if (publish_prediction && !samples_.empty())
        {
            const auto& latest = samples_.back();
            if (frame_ts_us <= latest.ts_us && latest.ts_us <= now_us &&
                now_us - latest.ts_us <= 300000)
                return latest;
        }
        return {frame_ts_us, frame_rotation};
    }

    bool empty() const { return samples_.empty(); }
    size_t size() const { return samples_.size(); }

private:
    size_t max_len_;
    std::deque<RotationSample> samples_;
};

#endif // ROTATION_HISTORY_H
