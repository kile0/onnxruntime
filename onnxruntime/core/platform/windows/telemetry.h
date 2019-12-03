// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/telemetry.h"
#include "core/platform/ort_mutex.h"
#include <atomic>

// ***
// platform specific control bits
#ifndef TraceLoggingOptionMicrosoftTelemetry
#define TraceLoggingOptionMicrosoftTelemetry() \
  TraceLoggingOptionGroup(0000000000, 00000, 00000, 0000, 0000, 0000, 0000, 0000, 000, 0000, 0000)
#endif
#define MICROSOFT_KEYWORD_MEASURES       0x0000400000000000  // Bit 46
#define TelemetryPrivacyDataTag(tag)     TraceLoggingUInt64((tag), "PartA_PrivTags")
#define PDT_ProductAndServicePerformance 0x0000000001000000u
#define PDT_ProductAndServiceUsage       0x0000000002000000u
// ***

namespace onnxruntime {

/**
  * derives and implments a Telemetry provider on Windows
  */
class WindowsTelemetry : public Telemetry {

 public:

  // these are allowed to be created, WindowsEnv will create one
  WindowsTelemetry();
  ~WindowsTelemetry();

  void EnableTelemetryEvents() const override;
  void DisableTelemetryEvents() const override;

  void LogProcessInfo() const override;

  void LogSessionCreation(uint32_t session_id, int64_t ir_version, const std::string& model_producer_name,
                          const std::string& model_producer_version, const std::string& model_domain,
                          const std::unordered_map<std::string, int>& domain_to_version_map,
                          const std::string& model_graph_name,
                          const std::unordered_map<std::string, std::string>& model_metadata,
                          const std::string& loadedFrom, const Vector<std::string>& execution_provider_ids) const override;
    
  void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file,
                       const char* function, uint32_t line) const override;

  void LogRuntimePerf(uint32_t session_id, uint32_t total_runs_since_last, int64_t total_run_duration_since_last) const override;

 private:
  static OrtMutex mutex_;
  static uint32_t global_register_count_;
  static bool enabled_;
};

}  // namespace onnxruntime
