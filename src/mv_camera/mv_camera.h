#pragma once 

#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <memory>
#include <functional>

#include <vpi/VPI.h>

#include "mv_camera_base.h"

class MVCamera : public MVCameraBase {
public:
    MVCamera(const std::string &device, int width, int height,
             int fps, bool triggerMode,
             int displayWidth, int displayHeight, 
             bool flip, bool mirror);

    virtual ~MVCamera();

    void setup(bool need_save, const std::string &save_path = ".", const std::string &filename_ts = "");

    virtual bool run(CallbackFn &&callback=nullptr) override;
    virtual bool run_async(CallbackFn &&callback=nullptr) override;
    virtual bool stop() override;
    virtual bool is_running() const override { return initialized_ && is_running_.load(); }
    virtual bool get_image_size(int &width, int &height) const override {
        width = displayWidth_;
        height = displayHeight_;
        return true;
    }
private:
    const std::string devicePath_;
    const int width_ = 0;
    const int height_ = 0;
    const int fps_ = 0;
    const bool triggerMode_ = false;
    const int displayWidth_ = 0;
    const int displayHeight_ = 0;

    int fd_ = -1;
    int pitch_ = 0;

    const char *i2c_dev;

    bool need_save_ = false;
    std::string save_path_ = ".";
    std::string filename_ts_;

    std::vector<std::pair<void*, size_t>> buffers_;
    std::deque<time_point> requestQueue_;
    std::mutex queueMutex_;
    std::atomic<bool> is_running_{false};
    std::thread triggerThread_;
    std::thread captureThread_;
    bool streaming_ = false;
    bool initialized_ = false;

    void *uv_ptr_ = nullptr;
    size_t uv_size_ = 0;

    void *jpegenc_ = nullptr;
    VPIStream vpiStream_ = nullptr;
    
    void triggerLoop(double interval_seconds);
    void init();
    void captureLoop(CallbackFn &&callback);
    void cleanup();
    uint16_t get_model();
    void set_image_flip_mirror(bool flip, bool mirror);
};