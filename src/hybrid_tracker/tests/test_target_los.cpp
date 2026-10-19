#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "rotation_history.h"
#include "rotation_kf.h"
#include "target_observation.h"

static int checks = 0;
static int failures = 0;

static void check(bool condition, const char* message)
{
    ++checks;
    if (!condition)
    {
        ++failures;
        std::printf("FAIL: %s\n", message);
    }
}

static bool close(double actual, double expected, double tolerance = 1e-10)
{
    return std::abs(actual - expected) <= tolerance;
}

static cv::Matx33d rotation(double yaw, double pitch, double roll)
{
    const double cy = std::cos(yaw), sy = std::sin(yaw);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cr = std::cos(roll), sr = std::sin(roll);
    return cv::Matx33d(cy, -sy, 0, sy, cy, 0, 0, 0, 1) *
           cv::Matx33d(cp, 0, sp, 0, 1, 0, -sp, 0, cp) *
           cv::Matx33d(1, 0, 0, 0, cr, -sr, 0, sr, cr);
}

static cv::Point2d pixel(const cv::Vec3d& ray)
{
    return {768 - 600 * ray[1] / ray[0], 480 - 600 * ray[2] / ray[0]};
}

static void testAnglesAndValidation()
{
    const auto identity = cv::Matx33d::eye();
    TargetLosEstimator estimator(600, 600, {768, 480});
    auto result = estimator.update({828, 540}, 1000000, identity);
    check(close(result.angles[0], std::atan2(.1, std::hypot(1., -.1))), "pitch down positive");
    check(close(result.angles[1], std::atan2(-.1, 1.)), "yaw left positive");
    check(cv::norm(result.rates) == 0, "first sample has zero angular rates");
    result = estimator.update({708, 480}, 1100000, identity);
    check(result.rates[1] > 0, "moving target has nonzero yaw rate");
    result = estimator.update({648, 480}, 2000000, identity);
    check(cv::norm(result.rates) == 0, "long gap clears velocity");
    for (int kind = 0; kind < 4; ++kind)
    {
        bool rejected = false;
        try
        {
            estimator.update(kind == 1 ? cv::Point2d(std::numeric_limits<double>::quiet_NaN(), 480) : cv::Point2d(768, 480),
                             kind == 0 ? 2000000 : 2100000,
                             kind == 2 ? identity * 2 : kind == 3 ? cv::Matx33d(-1, 0, 0, 0, 1, 0, 0, 0, 1) : identity);
        }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "reject repeated time, NaN, scaling and reflection");
    }
    estimator.reset();
    check(cv::norm(estimator.update({648, 480}, 500000, identity).rates) == 0, "reset accepts a new epoch");
    TargetLosEstimator anisotropic(600, 300, {768, 480}, 0);
    result = anisotropic.update({828, 510}, 1000000, identity);
    check(close(result.angles[0], std::atan2(.1, std::hypot(1., -.1))), "separate fx/fy support resized camera images");
    for (int kind = 0; kind < 4; ++kind)
    {
        bool rejected = false;
        try
        {
            TargetLosEstimator invalid(kind == 0 ? 0 : 600, kind == 1 ? -1 : 600,
                                       {768, 480}, kind == 2 ? -1 : .12, kind == 3 ? 0 : .3);
        }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "reject invalid estimator configuration");
    }
}

static void testCameraMotion()
{
    for (double tau : {0., .12})
    {
        TargetLosEstimator estimator(600, 600, {768, 480}, tau);
        const cv::Vec3d world_ray(1., .15, -.2);
        for (int i = 0; i < 60; ++i)
        {
            const auto camera = rotation(.2 * std::sin(i * .1), .3 * std::cos(i * .1), i * .02);
            const auto result = estimator.update(pixel(camera.t() * world_ray), 1000000 + i * 33333, camera);
            check(cv::norm(result.rates) < 1e-10, "stationary world target stays stationary through yaw/pitch/roll");
        }
    }
    TargetLosEstimator estimator(600, 600, {768, 480}, 0);
    cv::Vec3d previous_world_ray;
    for (int i = 0; i < 50; ++i)
    {
        const double t = i * .02, yaw = .2 * t, pitch = -.1 * t;
        const cv::Vec3d ray(std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), -std::sin(pitch));
        const auto camera = rotation(-.1 * t, .15 * t, .2 * t);
        const auto result = estimator.update(pixel(camera.t() * ray), 1000000 + i * 20000, camera);
        if (i > 0)
        {
            const cv::Vec3d old = camera.t() * previous_world_ray;
            check(close(result.rates[0], (result.angles[0] - std::atan2(-old[2], std::hypot(old[0], old[1]))) / .02),
                  "moving target pitch compensates camera rotation");
            check(close(result.rates[1], (result.angles[1] - std::atan2(old[1], old[0])) / .02),
                  "moving target yaw compensates camera rotation");
        }
        previous_world_ray = ray;
    }

    // Golden filtered rates from airsim_tracker/target_los.py at ab115d8.
    const double expected[][2] = {{0, 0}, {-.1359625571991271, .1258943168014166},
        {-.2480297877203754, .2345621446131098}, {-.3399263550801981, .3284307273525007},
        {-.4147990410745402, .409571065806772}};
    TargetLosEstimator filtered(600, 600, {768, 480});
    for (int i = 0; i < 5; ++i)
    {
        const auto result = filtered.update({768 + 2.25 * i, 480 - 1.75 * i}, 1000000 + i * 20000,
                                            rotation(i * .02, i * -.015, i * .01));
        check(close(result.rates[0], expected[i][0]) && close(result.rates[1], expected[i][1]),
              "filtered rate transport matches Python reference");
    }
}

static void testPredictionReference()
{
    RotationHistory history;
    const auto identity = cv::Matx33d::eye();
    const auto moved = rotation(.02, 0, 0);
    check(history.predictionReference(1000000, identity, 1100000, true).ts_us == 1000000, "missing pose falls back to capture time");
    history.push(1050000, moved);
    auto sample = history.predictionReference(1000000, identity, 1100000, false);
    check(sample.ts_us == 1000000 && cv::norm(sample.R - identity) == 0, "prediction disabled by default");
    sample = history.predictionReference(1000000, identity, 1100000, true);
    check(sample.ts_us == 1050000 && cv::norm(sample.R - moved) == 0, "prediction uses pose timestamp and matching rotation");
    check(history.predictionReference(1000000, identity, 1040000, true).ts_us == 1000000, "future pose rejected");
    check(history.predictionReference(1060000, identity, 1100000, true).ts_us == 1060000, "pose older than frame rejected");
    check(history.predictionReference(1000000, identity, 1350001, true).ts_us == 1000000, "stale pose rejected");
    check(history.predictionReference(1000000, identity, 1350000, true).ts_us == 1050000, "300ms freshness boundary accepted");
    history.push(1050000, identity);
    check(history.size() == 1, "duplicate pose ignored");
    history.push(500000, identity);
    check(history.size() == 1 && history.predictionReference(400000, identity, 600000, true).ts_us == 500000,
          "clock rollback starts a fresh rotation epoch");
}

static void testMessagesAndKalman()
{
    const auto identity = cv::Matx33d::eye();
    TargetObservationBuilder builder(600, 600, {768, 480}, .12);
    vision_msgs::msg::Detection2D detection;
    geometry_msgs::msg::TwistStamped los;
    cv::Rect2d box(758.5, 470.5, 20, 20);
    check(builder.build(&box, 123000000000, &identity, false, detection, los), "build paired observation");
    check(detection.header.stamp == los.header.stamp && los.header.frame_id == "camera_flu", "bbox and LOS share timestamp");
    check(close(detection.bbox.center.position.x, 768.5), "published bbox retains subpixel precision");
    check(close(los.twist.linear.z, std::atan2(-.5 / 600, 1)), "ROS yaw field follows camera FLU contract");
    check(los.twist.linear.x == 0 && los.twist.angular.x == 0, "unused ROS fields stay zero");
    check(!builder.build(&box, 123000000000, &identity, false, detection, los), "duplicate observation suppressed");
    check(!builder.build(&box, 122000000000, &identity, false, detection, los), "out-of-order observation suppressed");
    box.x -= 10;
    builder.build(&box, 123100000000, &identity, false, detection, los);
    check(los.twist.angular.z > 0, "ROS angular rate changes for a moving target");
    check(!builder.build(&box, 123100000000, &identity, true, detection, los),
          "reanchor at a repeated pose still suppresses duplicate output");
    builder.build(&box, 123200000000, &identity, false, detection, los);
    check(los.twist.angular.z == 0 && los.twist.angular.y == 0, "YOLO reanchor resets LOS rate even if output was skipped");
    builder.build(nullptr, 123300000000, nullptr, false, detection, los);
    check(detection.header.frame_id.empty() && los.header.frame_id.empty() && detection.header.stamp == los.header.stamp,
          "loss/OFF invalidates both messages together");
    builder.build(&box, 123400000000, &identity, false, detection, los);
    check(los.twist.angular.z == 0, "reacquisition starts with zero rate");
    builder.build(&box, 123500000000, nullptr, false, detection, los);
    check(!detection.header.frame_id.empty() && los.header.frame_id.empty(), "missing pose never produces a valid LOS");
    const auto invalid = identity * 2;
    builder.build(&box, 123600000000, &invalid, false, detection, los);
    check(los.header.frame_id.empty(), "invalid rotation invalidates LOS");
    builder.reset();
    check(builder.build(&box, 1000000000, &identity, false, detection, los), "ON resets publication timestamp guard");

    RotationKalmanFilter::Params params;
    params.fx = params.fy = 600;
    params.cx0 = 768;
    params.cy0 = 480;
    RotationKalmanFilter kalman(params);
    TargetLosEstimator estimator(600, 600, {768, 480}, 0);
    const cv::Vec3d world_ray(1., .1, -.15);
    for (int i = 0; i < 50; ++i)
    {
        const int64_t capture = 10000000 + i * 33333;
        const auto capture_rotation = rotation(i * .003, i * -.002, i * .004);
        const auto center = pixel(capture_rotation.t() * world_ray);
        kalman.update({center.x, center.y, 20, 20}, capture, capture_rotation);
        const auto publish_rotation = rotation(i * .003 + .001, i * -.002, i * .004);
        cv::Vec4d prediction;
        check(kalman.predictOnly(capture + 10000, publish_rotation, prediction), "KF prediction available");
        const auto result = estimator.update({prediction[0], prediction[1]}, capture + 10000, publish_rotation);
        check(cv::norm(result.rates) < 1e-4, "KF plus LOS preserves a stationary world ray");
    }
}

int main()
{
    testAnglesAndValidation();
    testCameraMotion();
    testPredictionReference();
    testMessagesAndKalman();
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
