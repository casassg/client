// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "concurrency_manager.h"
#include "periodic_concurrency_worker.h"

namespace triton { namespace perfanalyzer {

/// @brief Description
class PeriodicConcurrencyManager : public ConcurrencyManager {
 public:
  PeriodicConcurrencyManager(
      const bool async, const bool streaming, const int32_t batch_size,
      const size_t max_threads, const size_t max_concurrency,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const uint64_t concurrency_start, const uint64_t concurrency_step,
      const uint64_t request_period, const uint64_t num_steps)
      : ConcurrencyManager(
            async, streaming, batch_size, max_threads, max_concurrency,
            shared_memory_type, output_shm_size, parser, factory),
        concurrency_start_(concurrency_start),
        concurrency_step_(concurrency_step), request_period_(request_period),
        num_steps_(num_steps)
  {
  }

  void BeginConcurrency()
  {
    ChangeConcurrencyLevel(concurrency_start_);
    num_incomplete_periods_ = concurrency_start_;
  }

 private:
  std::shared_ptr<IWorker> MakeWorker(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<PeriodicConcurrencyWorker::ThreadConfig> thread_config)
      override
  {
    uint32_t id = workers_.size();

    auto worker = std::make_shared<PeriodicConcurrencyWorker>(
        id, thread_stat, thread_config, parser_, data_loader_, factory_,
        on_sequence_model_, async_, max_concurrency_, using_json_data_,
        streaming_, batch_size_, wake_signal_, wake_mutex_, active_threads_,
        execute_, infer_data_manager_, sequence_manager_, request_period_,
        manager_callback_);
    return worker;
  };

  void ManagerCallback(uint32_t worker_id)
  {
    std::lock_guard<std::mutex> lock(manager_callback_mutex_);

    if (--num_incomplete_periods_ == 0) {
      steps_completed_++;
      if (steps_completed_ < num_steps_) {
        AddConcurrentRequests(concurrency_step_);
        num_incomplete_periods_ = concurrency_step_;
      } else {
        Finalize();
      }
    }
  }

  void AddConcurrentRequests(uint64_t num_concurrent_requests)
  {
    for (size_t i = 0; i < num_concurrent_requests; i++) {
      threads_stat_.emplace_back(std::make_shared<ThreadStat>());
      threads_config_.emplace_back(
          std::make_shared<ConcurrencyWorker::ThreadConfig>(
              threads_config_.size(), 1, i));
      workers_.emplace_back(
          MakeWorker(threads_stat_.back(), threads_config_.back()));
      threads_.emplace_back(&IWorker::Infer, workers_.back());
      active_threads_++;
    }
  }

  void Finalize()
  {
    // CollectData();
  }

  uint64_t concurrency_start_{0};
  uint64_t concurrency_step_{0};
  uint64_t request_period_{0};
  uint64_t num_steps_{0};
  uint64_t steps_completed_{0};
  uint64_t num_incomplete_periods_{0};
  std::mutex manager_callback_mutex_{};
  std::function<void(uint32_t)> manager_callback_{std::bind(
      &PeriodicConcurrencyManager::ManagerCallback, this,
      std::placeholders::_1)};
};

}}  // namespace triton::perfanalyzer
