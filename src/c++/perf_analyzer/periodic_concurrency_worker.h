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

#include "concurrency_worker.h"

namespace triton { namespace perfanalyzer {

/// @brief
class PeriodicConcurrencyWorker : public ConcurrencyWorker {
 public:
  PeriodicConcurrencyWorker(
      uint32_t id, std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config,
      const std::shared_ptr<ModelParser> parser,
      std::shared_ptr<DataLoader> data_loader,
      const std::shared_ptr<cb::ClientBackendFactory> factory,
      const bool on_sequence_model, const bool async,
      const size_t max_concurrency, const bool using_json_data,
      const bool streaming, const int32_t batch_size,
      std::condition_variable& wake_signal, std::mutex& wake_mutex,
      size_t& active_threads, bool& execute,
      const std::shared_ptr<IInferDataManager>& infer_data_manager,
      std::shared_ptr<SequenceManager> sequence_manager,
      uint64_t request_period, std::function<void(uint32_t)> manager_callback)
      : ConcurrencyWorker(
            id, thread_stat, thread_config, parser, data_loader, factory,
            on_sequence_model, async, max_concurrency, using_json_data,
            streaming, batch_size, wake_signal, wake_mutex, active_threads,
            execute, infer_data_manager, sequence_manager),
        request_period_(request_period), manager_callback_(manager_callback)
  {
  }

  void Infer() override
  {
    CreateCtxIdTracker();
    ReserveContexts();
    RunInference();
  }

  std::shared_ptr<InferContext> CreateInferContext() override
  {
    std::shared_ptr infer_context{std::make_shared<InferContext>(
        id_, ctxs_.size(), async_, streaming_, on_sequence_model_,
        using_json_data_, batch_size_, thread_stat_, data_loader_, parser_,
        factory_, execute_, infer_data_manager_, sequence_manager_)};
    infer_context->RegisterWorkerCallback(worker_callback_);
    return infer_context;
  }


  void WorkerCallback(uint32_t infer_context_id)
  {
    if (ctxs_.at(infer_context_id)->GetNumResponsesForCurrentRequest() ==
        request_period_) {
      manager_callback_(id_);
    }
  }

 private:
  uint64_t request_period_{0};
  std::function<void(uint32_t)> manager_callback_{nullptr};
  std::function<void(uint32_t)> worker_callback_{std::bind(
      &PeriodicConcurrencyWorker::WorkerCallback, this, std::placeholders::_1)};
};

}}  // namespace triton::perfanalyzer
