// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/framework/ml_value.h"

namespace onnxruntime {
class ExecutionProviders;
class IExecutionProvider;
class OrtValueNameIdxMap;
class SessionState;

enum class DeviceCopyCheck {
  Unknown,
  NoCopy,
  Copy
};

struct DeviceCopyChecks {
  DeviceCopyCheck status = DeviceCopyCheck::Unknown;  ///< Overall status. If NoCopy no input or output copies are needed
  DeviceCopyCheck input_copy_needed = DeviceCopyCheck::Unknown;
  DeviceCopyCheck output_copy_needed = DeviceCopyCheck::Unknown;
};

struct FeedsFetchesInfo {
  FeedsFetchesInfo() = default;
  FeedsFetchesInfo(const Vector<std::string>& feed_names_in,
                   const Vector<std::string>& output_names_in,
                   const OrtValueNameIdxMap& ort_value_name_idx_map)
      : feed_names{feed_names_in}, output_names{output_names_in} {
    ORT_THROW_IF_ERROR(SetMLValueIdxs(ort_value_name_idx_map));
  }

  static Status MapNamesToMLValueIdxs(const Vector<std::string>& names,
                                      const OrtValueNameIdxMap& ort_value_name_idx_map,
                                      Vector<int>& ort_value_idxs);

  // set the ort_value_idxs for the current values in feed_names and output_names
  Status SetMLValueIdxs(const OrtValueNameIdxMap& ort_value_name_idx_map);

  Vector<std::string> feed_names;
  Vector<std::string> output_names;

  Vector<int> feeds_mlvalue_idxs;
  Vector<int> fetches_mlvalue_idxs;
};

struct MLValueCopyInfo {
  OrtDevice source_device{};
  OrtDevice target_device{};
  const IExecutionProvider* allocation_provider{nullptr};
};

class FeedsFetchesManager {
 public:
  static Status Create(const Vector<std::string>& feed_names, const Vector<std::string>& output_names,
                       const OrtValueNameIdxMap& ort_value_name_idx_map,
                       std::unique_ptr<FeedsFetchesManager>& feeds_fetches_manager);

  FeedsFetchesManager(FeedsFetchesInfo&& info);

  const FeedsFetchesInfo& GetFeedsFetchesInfo() const { return feeds_fetches_info_; }

  Vector<MLValueCopyInfo>& GetMutableFeedsDeviceCopyInfo() { return feeds_device_copy_info_; }
  const Vector<MLValueCopyInfo>& GetFeedsDeviceCopyInfo() const { return feeds_device_copy_info_; }

  Vector<MLValueCopyInfo>& GetMutableFetchesDeviceCopyInfo() { return fetches_device_copy_info_; }
  const Vector<MLValueCopyInfo>& GetFetchesDeviceCopyInfo() const { return fetches_device_copy_info_; }

  const DeviceCopyChecks& GetDeviceCopyChecks() const { return device_copy_checks_; }
  void SetDeviceCopyChecks(DeviceCopyCheck input_copy_needed, DeviceCopyCheck output_copy_needed);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FeedsFetchesManager);

  DeviceCopyChecks device_copy_checks_ = {};

  FeedsFetchesInfo feeds_fetches_info_;

  Vector<MLValueCopyInfo> feeds_device_copy_info_;
  Vector<MLValueCopyInfo> fetches_device_copy_info_;
};
}  // namespace onnxruntime
