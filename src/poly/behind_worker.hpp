#pragma once

#include "cpu_placement.hpp"

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace strata::poly::detail {

class BehindWorker {
public:
    BehindWorker() : thread_([this] { loop(); }) {}
    BehindWorker(const BehindWorker&) = delete;
    BehindWorker& operator=(const BehindWorker&) = delete;
    ~BehindWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            quit_ = true;
        }
        changed_.notify_one();
        thread_.join();
    }

    void go(std::function<void()> work) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!done_) throw std::logic_error("pipeline worker is already running");
            work_ = std::move(work);
            error_ = nullptr;
            done_ = false;
        }
        changed_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] { return done_; });
        const std::exception_ptr error = std::exchange(error_, nullptr);
        lock.unlock();
        if (error) std::rethrow_exception(error);
    }

private:
    void loop() {
        cpu_placement::pin_aux();
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            changed_.wait(lock, [this] { return quit_ || !done_; });
            if (done_ && quit_) return;
            auto work = std::move(work_);
            work_ = {};
            lock.unlock();
            std::exception_ptr error;
            try { work(); }
            catch (...) { error = std::current_exception(); }
            work = {};  // drop the Engine capture before wait() can return
            lock.lock();
            error_ = std::move(error);
            done_ = true;
            changed_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable changed_;
    std::function<void()> work_;
    std::exception_ptr error_;
    bool done_ = true, quit_ = false;
    std::thread thread_;
};

}  // namespace strata::poly::detail
