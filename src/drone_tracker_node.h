#pragma once
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <mavros_msgs/msg/status_text.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/int8.hpp>

#include <queue>
#include <thread>
#include <mutex>
#include <tuple>
#include <rosbag2_cpp/writer.hpp>

#include "hybrid_tracker.h"
#include "rotation_history.h"
#include "target_observation.h"
#include "mv_camera.h"
#include "mv_video.h"

// 从 mv_camera.* 和实际输出分辨率推导的内参，供旋转补偿 KF 和 LOS 共用。
struct CameraIntrinsics {
    double fx, fy, px, py;
};

class StatusPublisher
{
public:
    StatusPublisher(rclcpp::Node &node, int hz=1);
    ~StatusPublisher();
    bool publish(uint8_t severity, int count,const std::string& text, bool force=false);
private:
    rclcpp::Publisher<mavros_msgs::msg::StatusText>::SharedPtr status_pub_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    std::mutex status_mtx_;
    std::queue<std::tuple<uint8_t, int, std::string>> status_queue_;
    int status_counter_ = -1;
};
class DroneTrackerNode : public rclcpp::Node
{
public:
    DroneTrackerNode();
    ~DroneTrackerNode();

    void run();
    void stop();

private:
    void declare_parameters() {
        this->declare_parameter<std::string>("video_source", "0");
        this->declare_parameter<bool>("need_save", false);
        this->declare_parameter<std::string>("save_path", ".");
        this->declare_parameter<bool>("visualized_result", false);
        this->declare_parameter<bool>("use_history_timestamp", false);
        this->declare_parameter<bool>("publish_prediction", false);
        this->declare_parameter<bool>("enable_preview", true);
        this->declare_parameter<double>("los.rate_filter_tau", 0.12);

        this->declare_parameter<std::string>("tracker.yolo_engine", "drone_int8.engine");
        this->declare_parameter<std::string>("tracker.nano_backbone", "nanotrackv3_backbone_int8.engine");
        this->declare_parameter<std::string>("tracker.nano_head", "nanotrack_head_int8.engine");

        this->declare_parameter<int>("tracker.yolo_target_class_id", 0);
        this->declare_parameter<float>("tracker.yolo_conf_threshold", 0.4f);
        this->declare_parameter<float>("tracker.yolo_nms_threshold", 0.4f);
        this->declare_parameter<int>("tracker.yolo_interval", 4);
        this->declare_parameter<float>("tracker.yolo_recall_relax_ratio", 1.0f);
        this->declare_parameter<float>("tracker.iou_reset_threshold", 0.4f);
        this->declare_parameter<float>("tracker.iou_calibrate_threshold", 0.8f);
        this->declare_parameter<float>("tracker.max_position_jump", 150.0f);
        this->declare_parameter<float>("tracker.max_size_jump_ratio", 2.0f);
        this->declare_parameter<float>("tracker.max_box_area_ratio", 0.8f);
        this->declare_parameter<float>("tracker.tracking_roi_scale_factor", 2.0f);
        this->declare_parameter<int>("tracker.lost_max_count", 5);
        this->declare_parameter<int>("tracker.frame_buffer_size", 30);
        this->declare_parameter<int>("tracker.jump_tolerance", 3);
        this->declare_parameter<bool>("tracker.enable_dcf", true);
        this->declare_parameter<bool>("tracker.dcf_fixed_template", true);
        this->declare_parameter<double>("tracker.dcf_learning_rate", 0.01);
        this->declare_parameter<double>("tracker.dcf_lambda", 0.0001);
        this->declare_parameter<double>("tracker.dcf_psr_thresh", 5.0);
        this->declare_parameter<double>("tracker.dcf_max_div_factor", 0.5);
        this->declare_parameter<double>("tracker.kf_r_pos", 4.0);
        this->declare_parameter<double>("tracker.kf_r_size", 4.0);
        this->declare_parameter<double>("tracker.kf_q_a", 5e4);
        this->declare_parameter<double>("tracker.kf_q_size", 1e3);

        this->declare_parameter<int>("mv_camera.width_px", 1920);
        this->declare_parameter<int>("mv_camera.height_px", 1920);
        this->declare_parameter<bool>("mv_camera.flip", false);
        this->declare_parameter<bool>("mv_camera.mirror", false);
        this->declare_parameter<float>("mv_camera.pixel_size_um", 3.45f);
        this->declare_parameter<int>("mv_camera.fps", 30);
        this->declare_parameter<int>("mv_camera.output_width_px", 640);
        this->declare_parameter<int>("mv_camera.output_height_px", 640);
        this->declare_parameter<float>("mv_camera.focal_length_mm", 12.0f);
    }

    void setup();
    void worker();
    void start_preview();
    void stop_preview();
    void preview_loop();
    bool read_latest_frame(std::shared_ptr<VPIImageWrapper>& frame);
    bool decode_frame(const std::shared_ptr<VPIImageWrapper>& buffer, cv::Mat& frame);
    void publish_image(const cv::Mat& frame, int64_t timestamp_us);
    void publish_detections(const cv::Rect2d* box, int64_t timestamp_us,
                            const cv::Matx33d* rotation = nullptr, bool reset_los = false);

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void mavros_imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gimbal_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr mavros_imu_sub_;
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr trigger_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr detection_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr los_pub_;
    std::unique_ptr<TargetObservationBuilder> observation_builder_;
    rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr tracker_state_pub_;

    std::unique_ptr<rosbag2_cpp::Writer> rosbag_writer_;
    std::unique_ptr<StatusPublisher> status_publisher_;

    std::shared_ptr<HybridTracker> tracker_;
    std::shared_ptr<MVCameraBase> mv_camera_;

    int interval_ms_ = 1;
    bool enable_image_output_ = false;
    bool use_history_timestamp_ = false;
    bool publish_prediction_ = false;
    bool enable_preview_ = true;

    CameraIntrinsics cam_intrinsics_;

    // 云台旋转历史：(ts, R_cam)，为跟踪器 KF 提供帧时刻/当前时刻的相机旋转。
    // 订阅回调在触发 ON 之前就持续积累，保证启动时历史已就绪。
    RotationHistory gimbal_history_;
    mutable std::mutex gimbal_mtx_;

    std::thread worker_thread_;
    std::thread preview_thread_;
    std::atomic_bool preview_running_ = false;
    std::atomic_bool running = false;
    int8_t last_tracker_state = -1;
};