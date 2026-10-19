#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <opencv2/core.hpp>

// Camera FLU: pitch is positive down, yaw is positive left; radians and rad/s.
struct TargetLos
{
    cv::Vec2d angles{};
    cv::Vec2d rates{};
};

class TargetLosEstimator
{
public:
    TargetLosEstimator(double fx, double fy, cv::Point2d center,
                       double rate_tau = 0.12, double max_gap = 0.3)
        : fx_(fx), fy_(fy), center_(center), rate_tau_(rate_tau), max_gap_(max_gap)
    {
        if (!std::isfinite(fx) || fx <= 0 || !std::isfinite(fy) || fy <= 0 ||
            !std::isfinite(center.x) || !std::isfinite(center.y) ||
            !std::isfinite(rate_tau) || rate_tau < 0 ||
            !std::isfinite(max_gap) || max_gap <= 0)
            throw std::invalid_argument("Invalid LOS intrinsics or filtering parameters");
    }

    void reset()
    {
        has_previous_ = false;
        rate_ = cv::Vec2d();
    }

    TargetLos update(cv::Point2d center, int64_t timestamp_us, const cv::Matx33d& rotation)
    {
        if (!std::isfinite(center.x) || !std::isfinite(center.y) || !validRotation(rotation))
            throw std::invalid_argument("Invalid LOS sample or camera rotation");
        const cv::Vec3d ray(1.0, -(center.x - center_.x) / fx_, -(center.y - center_.y) / fy_);
        const cv::Vec2d angles = losAngles(ray);
        cv::Vec2d next_rate = rate_;
        if (has_previous_)
        {
            if (timestamp_us <= previous_time_us_)
                throw std::invalid_argument("LOS timestamps must increase");
            const double dt = (timestamp_us - previous_time_us_) * 1e-6;
            if (dt <= max_gap_)
            {
                const cv::Matx33d relative = rotation.t() * previous_rotation_;
                const cv::Vec3d old_ray = relative * previous_ray_;
                const cv::Vec2d old_angles = losAngles(old_ray);
                const cv::Vec3d old_velocity = relative * rayVelocity(losAngles(previous_ray_), rate_);
                next_rate = angularVelocity(old_ray / cv::norm(old_ray), old_velocity);
                const double alpha = rate_tau_ == 0 ? 1.0 : -std::expm1(-dt / rate_tau_);
                for (int i = 0; i < 2; ++i)
                {
                    const double delta = std::atan2(std::sin(angles[i] - old_angles[i]),
                                                    std::cos(angles[i] - old_angles[i]));
                    next_rate[i] += alpha * (delta / dt - next_rate[i]);
                }
            }
            else
                next_rate = cv::Vec2d();
        }
        previous_time_us_ = timestamp_us;
        previous_ray_ = ray;
        previous_rotation_ = rotation;
        rate_ = next_rate;
        has_previous_ = true;
        return {angles, rate_};
    }

private:
    static bool validRotation(const cv::Matx33d& rotation)
    {
        for (double value : rotation.val)
            if (!std::isfinite(value))
                return false;
        return cv::norm(rotation.t() * rotation - cv::Matx33d::eye(), cv::NORM_INF) <= 1e-5 &&
               std::abs(cv::determinant(rotation) - 1.0) <= 1e-5;
    }

    static cv::Vec2d losAngles(const cv::Vec3d& ray)
    {
        return {std::atan2(-ray[2], std::hypot(ray[0], ray[1])), std::atan2(ray[1], ray[0])};
    }

    static cv::Vec3d rayVelocity(const cv::Vec2d& angles, const cv::Vec2d& rates)
    {
        const double cp = std::cos(angles[0]), sp = std::sin(angles[0]);
        const double cy = std::cos(angles[1]), sy = std::sin(angles[1]);
        return rates[0] * cv::Vec3d(-sp * cy, -sp * sy, -cp) +
               rates[1] * cv::Vec3d(-cp * sy, cp * cy, 0);
    }

    static cv::Vec2d angularVelocity(const cv::Vec3d& ray, const cv::Vec3d& velocity)
    {
        const double x = ray[0], y = ray[1], z = ray[2];
        const double dx = velocity[0], dy = velocity[1], dz = velocity[2];
        const double horizontal = std::hypot(x, y);
        if (horizontal < 1e-6)
            throw std::invalid_argument("LOS angular rates are undefined at the vertical pole");
        return {(-horizontal * dz + z * (x * dx + y * dy) / horizontal) / ray.dot(ray),
                (x * dy - y * dx) / (horizontal * horizontal)};
    }

    double fx_, fy_;
    cv::Point2d center_;
    double rate_tau_, max_gap_;
    bool has_previous_ = false;
    int64_t previous_time_us_ = 0;
    cv::Vec3d previous_ray_{};
    cv::Matx33d previous_rotation_ = cv::Matx33d::eye();
    cv::Vec2d rate_{};
};
