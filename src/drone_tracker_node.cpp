#include "drone_tracker_node.h"

#include <unordered_map>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std::chrono_literals;

static const std::string TOPIC_GIMBAL = "/fpv/gimbal";
static const std::string TOPIC_MAVROS_IMU = "/mavros/imu/data";
static const std::string TOPIC_ODOM = "/fpv/odom";
static const std::string TOPIC_TARGET_BBOX = "/fpv/target_bbox";
static const std::string TOPIC_TARGET_LOS = "/fpv/target_los";
static const std::string TOPIC_TRACKER_IMAGE = "/fpv/tracker_image";
static const std::string TOPIC_TRACKER_STATE =  "/fpv/tracker_state";
static const std::string TOPIC_TRACKER_TRIGGER = "/fpv/tracker_trigger";

DroneTrackerNode::DroneTrackerNode()
: Node("drone_tracker_node")
{
    declare_parameters();
    setup();
}

DroneTrackerNode::~DroneTrackerNode()
{
    stop_preview();
    stop();
    if (rosbag_writer_)
        rosbag_writer_->close();
}

void DroneTrackerNode::run()
{
    if (running) return;
    stop_preview();
    if (worker_thread_.joinable())
        worker_thread_.join();
    tracker_->reset();
    observation_builder_->reset();
    last_tracker_state = -1;
    running = true;
    worker_thread_ = std::thread(&DroneTrackerNode::worker, this);
}

void DroneTrackerNode::stop()
{
    running = false;
    if (worker_thread_.joinable())
    {
        worker_thread_.join();
    }
}

void DroneTrackerNode::setup()
{
    // Initialize publishers, subscribers, and services here
    this->get_parameter("visualized_result", enable_image_output_);
    this->get_parameter("use_history_timestamp", use_history_timestamp_);
    this->get_parameter("publish_prediction", publish_prediction_);
    this->get_parameter("enable_preview", enable_preview_);

    int width_px, height_px, output_width_px, output_height_px, fps;
    float pixel_size_um, focal_length_mm;
    bool flip, mirror;
    this->get_parameter("mv_camera.width_px", width_px);
    this->get_parameter("mv_camera.height_px", height_px);
    this->get_parameter("mv_camera.flip", flip);
    this->get_parameter("mv_camera.mirror", mirror);
    this->get_parameter("mv_camera.focal_length_mm", focal_length_mm);
    this->get_parameter("mv_camera.pixel_size_um", pixel_size_um);
    this->get_parameter("mv_camera.fps", fps);
    this->get_parameter("mv_camera.output_width_px", output_width_px);
    this->get_parameter("mv_camera.output_height_px", output_height_px);

    std::string video_source;
    this->get_parameter("video_source", video_source);
    const bool is_camera = (video_source.substr(0, 5) == "video" && video_source.length() == 6);

    if (is_camera)
    {
        RCLCPP_INFO(this->get_logger(), "Opening camera device: %s", video_source.c_str());
        std::string device = "/dev/";
        mv_camera_ = std::make_shared<MVCamera>(device.append(video_source), 
            width_px, height_px, fps, true, output_width_px, output_height_px, flip, mirror);
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Opening video file: %s", video_source.c_str());
        mv_camera_ = std::make_shared<MVVideo>(video_source, use_history_timestamp_);
    }

    mv_camera_->get_image_size(output_width_px, output_height_px);

    // Assuming square pixels for simplicity
    float scale_ratio_w = static_cast<float>(output_width_px) / static_cast<float>(width_px);
    float scale_ratio_h = static_cast<float>(output_height_px) / static_cast<float>(height_px);

    cam_intrinsics_.fx = focal_length_mm * 1000.0 / pixel_size_um * scale_ratio_w;
    cam_intrinsics_.fy = focal_length_mm * 1000.0 / pixel_size_um * scale_ratio_h;
    cam_intrinsics_.px = output_width_px / 2.0;
    cam_intrinsics_.py = output_height_px / 2.0;
    double rate_tau;
    this->get_parameter("los.rate_filter_tau", rate_tau);
    observation_builder_ = std::make_unique<TargetObservationBuilder>(
        cam_intrinsics_.fx, cam_intrinsics_.fy,
        cv::Point2d(cam_intrinsics_.px, cam_intrinsics_.py), rate_tau);

    double fov_deg_w = std::atan2(output_width_px * 0.5, cam_intrinsics_.fx) * 180.0 / M_PI * 2.0;
    double fov_deg_h = std::atan2(output_height_px * 0.5, cam_intrinsics_.fy) * 180.0 / M_PI * 2.0;

    RCLCPP_INFO(this->get_logger(), "Camera Intrinsics: fx=%.2f, fy=%.2f, px=%.2f, py=%.2f", 
        cam_intrinsics_.fx, cam_intrinsics_.fy, cam_intrinsics_.px, cam_intrinsics_.py);
    RCLCPP_INFO(this->get_logger(), "Camera FOV: horizontal=%.2f deg, vertical=%.2f deg", fov_deg_w, fov_deg_h);

    TrackerSettings tracker_settings;

    std::string share_dir = ament_index_cpp::get_package_share_directory(ROS_PACKAGE_NAME);
    std::cout << "Package share directory: " << share_dir << std::endl;

    this->get_parameter("tracker.yolo_engine", tracker_settings.yolo_engine_path);
    this->get_parameter("tracker.nano_backbone", tracker_settings.nanotrack_backbone_path);
    this->get_parameter("tracker.nano_head", tracker_settings.nanotrack_head_path);
    this->get_parameter("tracker.yolo_target_class_id", tracker_settings.target_class_id);
    this->get_parameter("tracker.yolo_conf_threshold", tracker_settings.yolo_conf_threshold);
    this->get_parameter("tracker.yolo_nms_threshold", tracker_settings.yolo_nms_threshold);
    this->get_parameter("tracker.yolo_interval", tracker_settings.yolo_interval);
    this->get_parameter("tracker.iou_reset_threshold", tracker_settings.iou_reset_threshold);
    this->get_parameter("tracker.iou_calibrate_threshold", tracker_settings.iou_calibrate_threshold);
    this->get_parameter("tracker.yolo_recall_relax_ratio", tracker_settings.yolo_recall_relax_ratio);
    this->get_parameter("tracker.max_position_jump", tracker_settings.max_position_jump);
    this->get_parameter("tracker.max_size_jump_ratio", tracker_settings.max_size_jump_ratio);
    this->get_parameter("tracker.max_box_area_ratio", tracker_settings.max_box_area_ratio);
    this->get_parameter("tracker.lost_max_count", tracker_settings.lost_max_count);
    this->get_parameter("tracker.tracking_roi_scale_factor", tracker_settings.roi_scale_factor);
    this->get_parameter("tracker.frame_buffer_size", tracker_settings.frame_buffer_size);
    this->get_parameter("tracker.jump_tolerance", tracker_settings.jump_tolerance);
    this->get_parameter("tracker.enable_dcf", tracker_settings.enable_dcf);
    this->get_parameter("tracker.dcf_fixed_template", tracker_settings.dcf.fixed_template);
    this->get_parameter("tracker.dcf_learning_rate", tracker_settings.dcf.learning_rate);
    this->get_parameter("tracker.dcf_lambda", tracker_settings.dcf.lambda_reg);
    this->get_parameter("tracker.dcf_psr_thresh", tracker_settings.dcf.psr_thresh);
    this->get_parameter("tracker.dcf_max_div_factor", tracker_settings.dcf_max_div_factor);
    this->get_parameter("tracker.kf_r_pos", tracker_settings.kf_r_pos);
    this->get_parameter("tracker.kf_r_size", tracker_settings.kf_r_size);
    this->get_parameter("tracker.kf_q_a", tracker_settings.kf_q_a);
    this->get_parameter("tracker.kf_q_size", tracker_settings.kf_q_size);

    // KF 内参来自相机参数（Python 端 camera.fov_deg 家族的等价物）
    tracker_settings.kf_fx = cam_intrinsics_.fx;
    tracker_settings.kf_fy = cam_intrinsics_.fy;
    tracker_settings.kf_cx0 = cam_intrinsics_.px;
    tracker_settings.kf_cy0 = cam_intrinsics_.py;

    tracker_settings.yolo_engine_path = share_dir + "/models/" + tracker_settings.yolo_engine_path;
    tracker_settings.nanotrack_backbone_path = share_dir + "/models/" + tracker_settings.nanotrack_backbone_path;
    tracker_settings.nanotrack_head_path = share_dir + "/models/" + tracker_settings.nanotrack_head_path;

    tracker_ = HybridTracker::CreateInstance(tracker_settings);
    if (!tracker_->init())
    {
        throw std::runtime_error("Failed to initialize HybridTracker");
    }

    gimbal_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        TOPIC_GIMBAL, rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
        std::bind(&DroneTrackerNode::imu_callback, this, std::placeholders::_1)
    );

    mavros_imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        TOPIC_MAVROS_IMU, rclcpp::QoS(rclcpp::KeepLast(100)).best_effort(),
        std::bind(&DroneTrackerNode::mavros_imu_callback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        TOPIC_ODOM, 
        rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(), 
        std::bind(&DroneTrackerNode::odom_callback, this, std::placeholders::_1));

    detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2D>(
        TOPIC_TARGET_BBOX, 10);
    los_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
        TOPIC_TARGET_LOS, 10);
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        TOPIC_TRACKER_IMAGE, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort());
    tracker_state_pub_ = this->create_publisher<std_msgs::msg::Int8>(
        TOPIC_TRACKER_STATE, 10);

    status_publisher_ = std::make_unique<StatusPublisher>(*this, 2);
    
    trigger_sub_ = this->create_subscription<std_msgs::msg::Header>(
        TOPIC_TRACKER_TRIGGER, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
        [this](const std_msgs::msg::Header::SharedPtr msg)
        {
            if (msg->frame_id == "ON" && !running)
            {
                stop_preview();
                stop();
                if (rosbag_writer_)
                {
                    rosbag_writer_->close();
                    rosbag_writer_.reset();
                }
                // Generate a timestamped filename prefix
                std::time_t t = msg->stamp.sec;
                std::tm* tm_ptr = std::localtime(&t);
                char time_buffer[32];
                std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", tm_ptr);
                long microseconds = msg->stamp.nanosec / 1000;
                std::string filename_ts = std::string(time_buffer) + "_" + std::to_string(microseconds);
                
                bool need_save = false;
                this->get_parameter("need_save", need_save);
                std::string save_path;
                this->get_parameter("save_path", save_path);
                
                if (mv_camera_)
                {
                    auto camera = std::dynamic_pointer_cast<MVCamera>(mv_camera_);
                    if (camera)
                        camera->setup(need_save, save_path, filename_ts);
                }
                if (need_save)
                {
                    rosbag_writer_ = std::make_unique<rosbag2_cpp::Writer>();
                    rosbag_writer_->open(save_path + "/drone_rosbag_" + filename_ts);
                    rosbag2_storage::TopicMetadata topic;
                    topic.name = TOPIC_ODOM;
                    topic.type = "nav_msgs/msg/Odometry";
                    topic.serialization_format = rmw_get_serialization_format();
                    rosbag_writer_->create_topic(topic);
                    topic.name = TOPIC_TARGET_LOS;
                    topic.type = "geometry_msgs/msg/TwistStamped";
                    rosbag_writer_->create_topic(topic);
                    topic.name = TOPIC_MAVROS_IMU;
                    topic.type = "sensor_msgs/msg/Imu";
                    rosbag_writer_->create_topic(topic);
                }

                this->run();
                status_publisher_->publish(mavros_msgs::msg::StatusText::WARNING, 1, "Drone Tracker Started", true);
            }
            else if (msg->frame_id == "OFF" && (running || worker_thread_.joinable()))
            {
                this->stop();
                const int64_t stamp_us = std::max(
                    this->now().nanoseconds() / 1000, observation_builder_->lastStamp() / 1000 + 1);
                publish_detections(nullptr, stamp_us);
                if (rosbag_writer_)
                {
                    rosbag_writer_->close();
                    rosbag_writer_.reset();
                }
                start_preview();
                status_publisher_->publish(mavros_msgs::msg::StatusText::WARNING, 1, "Drone Tracker Stopped", true);
            }
        }
    );
    start_preview();
}

void DroneTrackerNode::start_preview()
{
    // File replay must remain at its current position until ON.
    auto camera = std::dynamic_pointer_cast<MVCamera>(mv_camera_);
    if (!enable_preview_ || !camera || running || preview_running_)
        return;
    stop_preview();
    camera->setup(false);
    preview_running_ = true;
    preview_thread_ = std::thread(&DroneTrackerNode::preview_loop, this);
}

void DroneTrackerNode::stop_preview()
{
    preview_running_ = false;
    if (preview_thread_.joinable())
        preview_thread_.join();
}

bool DroneTrackerNode::read_latest_frame(std::shared_ptr<VPIImageWrapper>& frame)
{
    if (!mv_camera_->getImageQueue().wait_dequeue_timed(frame, 200ms))
        return false;
    std::shared_ptr<VPIImageWrapper> newer;
    while (mv_camera_->getImageQueue().try_dequeue(newer))
        frame = std::move(newer);
    return frame && frame->get();
}

bool DroneTrackerNode::decode_frame(const std::shared_ptr<VPIImageWrapper>& buffer, cv::Mat& frame)
{
    VpiGrayLock lock(buffer);
    if (!lock.valid() || lock.gray().empty())
        return false;
    cv::cvtColor(lock.gray(), frame, cv::COLOR_GRAY2BGR);
    return true;
}

void DroneTrackerNode::publish_image(const cv::Mat& frame, int64_t timestamp_us)
{
    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = rclcpp::Time(timestamp_us * 1000);
    cv_img.header.frame_id = "camera_frame";
    cv_img.encoding = "bgr8";
    cv_img.image = frame;
    auto msg = cv_img.toImageMsg();
    image_pub_->publish(*msg);
}

void DroneTrackerNode::publish_detections(const cv::Rect2d* box, int64_t timestamp_us,
                                         const cv::Matx33d* rotation, bool reset_los)
{
    vision_msgs::msg::Detection2D detection;
    geometry_msgs::msg::TwistStamped los;
    if (!observation_builder_->build(box, timestamp_us * 1000, rotation, reset_los, detection, los))
        return;
    if (box && los.header.frame_id.empty())
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Invalid target LOS or missing /fpv/gimbal camera rotation");
    if (rosbag_writer_)
    {
        rosbag_writer_->write(detection, TOPIC_TARGET_BBOX, detection.header.stamp);
        rosbag_writer_->write(los, TOPIC_TARGET_LOS, los.header.stamp);
    }
    detection_pub_->publish(detection);
    los_pub_->publish(los);
}

void DroneTrackerNode::preview_loop()
{
    std::shared_ptr<VPIImageWrapper> buffer;
    while (mv_camera_->getImageQueue().try_dequeue(buffer)) {}
    bool camera_started = false;
    while (rclcpp::ok() && preview_running_)
    {
        if (!camera_started)
        {
            camera_started = mv_camera_->run_async();
            if (!camera_started)
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                    "Preview camera unavailable; retrying");
                rclcpp::sleep_for(200ms);
                continue;
            }
        }
        if (!read_latest_frame(buffer))
            continue;
        if (image_pub_->get_subscription_count() == 0)
            continue;
        cv::Mat frame;
        if (decode_frame(buffer, frame))
            publish_image(frame, buffer->get_timestamps());
    }
    buffer.reset();
    if (camera_started)
        mv_camera_->stop();
    preview_running_ = false;
}

void DroneTrackerNode::worker()
{
    std::shared_ptr<VPIImageWrapper> stale;
    while (mv_camera_->getImageQueue().try_dequeue(stale)) {}
    stale.reset();
    bool success = mv_camera_->run_async();
    if (!success)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to open MV Camera ");
        running = false;
        return;
    }
    RCLCPP_INFO(this->get_logger(), "MV Camera opened successfully");

    static const std::unordered_map<int, std::string> state_text_map = {
        {0, "Detecting..."},
        {1, "Target Found"},
        {2, "Tracking..."},
        {3, "Target Lost"}
    };

    cv::Mat frame;

    int count = 0;
    int empty_reads = 0;
    double avg_time_ms = 0.0;

    // Main processing loop
    while (rclcpp::ok() && running)
    {
        rclcpp::Time start_time = this->now();
        std::shared_ptr<VPIImageWrapper> vpi_buffer;

        if (!read_latest_frame(vpi_buffer))
        {
            if (++empty_reads >= 5)
            {
                RCLCPP_WARN(this->get_logger(), "Camera stopped producing frames");
                break;
            }
            continue;
        }
        empty_reads = 0;
        if (!running)
            break;
        start_time = this->now();

        // 构造帧上下文：帧时刻与当前时刻的相机旋转（无云台数据时保持单位阵）
        FrameContext ctx;
        ctx.frame = vpi_buffer;
        ctx.frame_ts_us = vpi_buffer->get_timestamps();
        const int64_t now_us = this->now().nanoseconds() / 1000;
        bool has_camera_rotation;
        {
            std::lock_guard<std::mutex> lock(gimbal_mtx_);
            has_camera_rotation = gimbal_history_.nearest(ctx.frame_ts_us, ctx.R_cam_at_frame);
            const auto reference = gimbal_history_.predictionReference(
                ctx.frame_ts_us, ctx.R_cam_at_frame, now_us, publish_prediction_);
            ctx.now_us = reference.ts_us;
            ctx.R_cam_now = reference.R;
        }
        if (publish_prediction_ && ctx.now_us == ctx.frame_ts_us)
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "No newer fresh /fpv/gimbal pose; using capture-time bbox/LOS");

        TrackerOutput out = tracker_->update(ctx);

        auto elapsed = this->now() - start_time;
        double cost_ms = elapsed.seconds() * 1000;

        if (out.consumed)
        {
            const bool valid = out.tracking && out.observation_box.area() > 0;
            publish_detections(valid ? &out.observation_box : nullptr, ctx.now_us,
                               has_camera_rotation ? &ctx.R_cam_now : nullptr, out.reset_los);

            // 状态仅变化时发布
            int8_t tracker_state_int = static_cast<int8_t>(out.state);
            if (tracker_state_int != last_tracker_state)
            {
                auto it = state_text_map.find(tracker_state_int);
                status_publisher_->publish(mavros_msgs::msg::StatusText::WARNING, 1,
                    it != state_text_map.end() ? it->second : "Unknown", true);
                auto msg = std::make_unique<std_msgs::msg::Int8>();
                msg->data = tracker_state_int;
                tracker_state_pub_->publish(std::move(msg));
                last_tracker_state = tracker_state_int;
            }
        }

        if (enable_image_output_ && out.consumed &&
            (image_pub_->get_subscription_count() > 0 || rosbag_writer_))
        {
            const cv::Scalar color = cv::Scalar((out.color & 0xFF), ((out.color >> 8) & 0xFF), ((out.color >> 16) & 0xFF));
            const int64_t frame_ts = ctx.frame_ts_us;

            if (!decode_frame(vpi_buffer, frame))
                continue;

            if (out.display_box.is_valid)
            {
                cv::Rect display_box = {
                    out.display_box.cx - out.display_box.width / 2,
                    out.display_box.cy - out.display_box.height / 2,
                    out.display_box.width,
                    out.display_box.height };
                cv::rectangle(frame, display_box, color, 2, 1);
            }

            if (out.realtime_box.is_valid)
            {
                // 红色实时预测框
                cv::Point p1(out.realtime_box.cx - out.realtime_box.width / 2,
                             out.realtime_box.cy - out.realtime_box.height / 2);
                cv::Point p2(out.realtime_box.cx + out.realtime_box.width / 2,
                             out.realtime_box.cy + out.realtime_box.height / 2);
                cv::rectangle(frame, p1, p2, cv::Scalar(0, 0, 255), 2, 1);
                cv::putText(frame, "Real-Time Pred", cv::Point(p1.x, std::max(20, p1.y - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
            }

            cv::putText(frame, "Cost: " + std::to_string(int(cost_ms)) + " ms", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
            cv::putText(frame, out.info, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);

            publish_image(frame, frame_ts);

            elapsed = this->now() - start_time;
            cost_ms = elapsed.seconds() * 1000;
        }

        avg_time_ms = (avg_time_ms * count + cost_ms) / (count + 1);
        ++count;

        // 计算剩余时间以保持帧率
        int remaining_time_us = static_cast<int>((interval_ms_ - cost_ms) * 1000);
        if (remaining_time_us <= 0) remaining_time_us = 1000; // 至少等待1ms

        rclcpp::sleep_for(std::chrono::microseconds(remaining_time_us));
    }

    std::cout << "Average tracking time per frame: " << std::fixed << std::setprecision(2) << avg_time_ms << " ms" << std::endl;

    mv_camera_->stop();
    running = false;
}

void DroneTrackerNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (rosbag_writer_)
    {
        rosbag_writer_->write(*msg, TOPIC_ODOM, rclcpp::Time(msg->header.stamp));
    }
}

void DroneTrackerNode::mavros_imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    if (rosbag_writer_)
    {
        rosbag_writer_->write(*msg, TOPIC_MAVROS_IMU, rclcpp::Time(msg->header.stamp));
    }
}

void DroneTrackerNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    // /fpv/gimbal 携带相机到世界的姿态（FLU 约定，由 apm_bridge 发布）。
    // 跟踪器 KF 只使用相对旋转 R_new^T * R_old，世界系约定无关紧要。
    // 无论是否 running 都持续积累，保证触发 ON 时历史已就绪。
    const auto& q = msg->orientation;
    if (q.w == 0.0 && q.x == 0.0 && q.y == 0.0 && q.z == 0.0)
        return; // 无姿态数据

    const int64_t ts_us = rclcpp::Time(msg->header.stamp).nanoseconds() / 1000;

    // 四元数 -> 旋转矩阵（与 Eigen toRotationMatrix 同式），归一化后按行列展开
    const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (!std::isfinite(norm) || norm <= 0.0)
        return;
    const double w = q.w / norm, x = q.x / norm, y = q.y / norm, z = q.z / norm;
    const cv::Matx33d R(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));

    {
        std::lock_guard<std::mutex> lock(gimbal_mtx_);
        gimbal_history_.push(ts_us, R);
    }

    if (rosbag_writer_)
    {
        rosbag_writer_->write(*msg, TOPIC_GIMBAL, rclcpp::Time(msg->header.stamp));
    }
}

StatusPublisher::StatusPublisher(rclcpp::Node &node, int hz)
{
    status_pub_ = node.create_publisher<mavros_msgs::msg::StatusText>(
        "/mavros/statustext/send", 10);
    status_timer_ = node.create_wall_timer(std::chrono::milliseconds(1000 / hz), [this](){
        static uint8_t severity;
        static std::string status;

        std::lock_guard<std::mutex> lock(status_mtx_);
        if (status_counter_<=0)
        {
            if (status_queue_.empty()) return;
            auto [level, counter, text] = status_queue_.front();
            severity = level;
            status = text;
            status_counter_ = counter;
            // printf("Publishing status: severity=%d, count=%d, text=%s\n", severity, status_counter_, status.c_str());
            status_queue_.pop();
        }
        if (status_counter_-- > 0)
        {
            // Publish status
            auto msg = std::make_unique<mavros_msgs::msg::StatusText>();
            msg->header.stamp = rclcpp::Clock().now();
            msg->severity = severity; // mavros_msgs::msg::StatusText::WARNING;
            msg->text= status;
            status_pub_->publish(std::move(msg));
        }
    });
}

StatusPublisher::~StatusPublisher()
{
    if (status_timer_)
        status_timer_->cancel();
}

bool StatusPublisher::publish(uint8_t severity, int count,const std::string& text, bool force)
{
    std::lock_guard<std::mutex> lock(status_mtx_);
    if (!force && (status_counter_ > 0 || !status_queue_.empty()))
        return false;
    status_queue_.emplace(severity, count, text);
    return true;
}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DroneTrackerNode>();
    // node->run();
    rclcpp::spin(node);
    node.reset();
    rclcpp::shutdown();
    return 0;
}
