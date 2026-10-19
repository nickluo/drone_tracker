#pragma once

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <vision_msgs/msg/detection2_d.hpp>

#include "target_los.h"

// Build both messages together so their timestamps and invalidation stay paired.
class TargetObservationBuilder
{
public:
    TargetObservationBuilder(double fx, double fy, cv::Point2d center, double rate_tau)
        : estimator_(fx, fy, center, rate_tau) {}

    void reset()
    {
        estimator_.reset();
        last_stamp_ns_ = 0;
    }

    int64_t lastStamp() const { return last_stamp_ns_; }

    bool build(const cv::Rect2d* box, int64_t stamp_ns, const cv::Matx33d* rotation,
               bool reset_los, vision_msgs::msg::Detection2D& detection,
               geometry_msgs::msg::TwistStamped& los)
    {
        if (reset_los || !box)
            estimator_.reset();
        if (stamp_ns <= last_stamp_ns_)
            return false;
        last_stamp_ns_ = stamp_ns;
        detection = vision_msgs::msg::Detection2D();
        los = geometry_msgs::msg::TwistStamped();
        detection.header.stamp.sec = static_cast<int32_t>(stamp_ns / 1000000000);
        detection.header.stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000);
        los.header.stamp = detection.header.stamp;
        if (!box)
            return true;

        detection.header.frame_id = "camera_frame";
        const cv::Point2d center(box->x + box->width / 2.0, box->y + box->height / 2.0);
        detection.bbox.center.position.x = center.x;
        detection.bbox.center.position.y = center.y;
        detection.bbox.size_x = box->width;
        detection.bbox.size_y = box->height;
        if (!rotation)
        {
            estimator_.reset();
            return true;
        }
        try
        {
            const auto result = estimator_.update(center, stamp_ns / 1000, *rotation);
            los.header.frame_id = "camera_flu";
            los.twist.linear.y = result.angles[0];
            los.twist.linear.z = result.angles[1];
            los.twist.angular.y = result.rates[0];
            los.twist.angular.z = result.rates[1];
        }
        catch (const std::invalid_argument&)
        {
            estimator_.reset();
        }
        return true;
    }

private:
    TargetLosEstimator estimator_;
    int64_t last_stamp_ns_ = 0;
};
