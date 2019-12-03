// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License
#ifdef _MSC_VER
#pragma warning(disable : 4505)  //Unreferenced local function has been removed
#endif

#include "mkldnn_func_kernel.h"
#include "core/common/exceptions.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/mkldnn/subgraph/mkldnn_conv.h"
#include "core/providers/mkldnn/subgraph/mkldnn_batchnorm.h"
#include "core/providers/mkldnn/subgraph/mkldnn_conv_batchnorm.h"
#include "core/providers/mkldnn/subgraph/mkldnn_activations.h"
#include "core/providers/mkldnn/subgraph/mkldnn_pool.h"
#include "core/providers/mkldnn/subgraph/mkldnn_sum.h"
#include "core/providers/mkldnn/subgraph/mkldnn_lrn.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace mkl_dnn {

namespace {
template <typename T>
class SubgraphPrimitive : public PrimitiveBase {
 public:
  SubgraphPrimitive(const OrtCustomOpApi* api,
                    OrtKernelContext* context,
                    const SubgraphParams& params)
      : cpu_engine_(GetEngine()) {
    context_.stream = onnxruntime::make_unique<mkldnn::stream>(mkldnn::stream(cpu_engine_));

    if (context_.net.size() == 0) {
      CreateKernels(params);
      Initialize(api, context);
    }
  }

  void UpdateProvider(const SubgraphParams& params) {
    if (context_.kernels.size() > 0 && context_.kernels[0]->GetProvider() != params.provider)
      for (auto& kernel : context_.kernels) {
        kernel->SetProvider(params.provider);
      }
  }

  Status Compute(const OrtCustomOpApi* api, OrtKernelContext* context) {
    Status status;

    for (auto& kernel : context_.kernels) {
      ORT_RETURN_IF_ERROR(kernel->Bind(api, context));
    }
    for (size_t i = 0; i < context_.net.size(); ++i) {
      context_.net.at(i).execute(*context_.stream, context_.net_args.at(i));
    }
    return Status::OK();
  }

  ~SubgraphPrimitive() = default;

 private:
  void CreateKernels(const SubgraphParams& params) {
    for (const auto& mkldnn_node : params.subgraph->mkldnn_nodes) {
      if (mkldnn_node.name == "Conv") {
        std::ostringstream os;
        os << "Conv-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnConv<T>> kernel;
        kernel = std::make_shared<MklDnnConv<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "Conv-Relu") {
        std::ostringstream os;
        os << "Conv-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnConv<T>> kernel;
        kernel = std::make_shared<MklDnnConv<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        kernel->fuse_relu_ = true;
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "Relu") {
        std::ostringstream os;
        os << "Relu-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnRelu<T>> kernel;
        kernel = std::make_shared<MklDnnRelu<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "BatchNormalization") {
        std::ostringstream os;
        os << "BatchNormalization-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnBatchNorm<T>> kernel;
        kernel = std::make_shared<MklDnnBatchNorm<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "BatchNormalization-Relu") {
        std::ostringstream os;
        os << "BatchNormalization-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnBatchNorm<T>> kernel;
        kernel = std::make_shared<MklDnnBatchNorm<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        kernel->fuse_relu_ = true;
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "Conv-BatchNormalization") {
        std::ostringstream os;
        os << "Conv-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnConvBatchNorm<T>> kernel;
        kernel = std::make_shared<MklDnnConvBatchNorm<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "Conv-BatchNormalization-Relu") {
        std::ostringstream os;
        os << "Conv-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnConvBatchNorm<T>> kernel;
        kernel = std::make_shared<MklDnnConvBatchNorm<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        kernel->fuse_relu_ = true;
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "MaxPool") {
        std::ostringstream os;
        os << "MaxPool-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnPool<T>> kernel;
        kernel = std::make_shared<MklDnnPool<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "GlobalMaxPool") {
        std::ostringstream os;
        os << "GlobalMaxPool-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnPool<T>> kernel;
        kernel = std::make_shared<MklDnnPool<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "AveragePool") {
        std::ostringstream os;
        os << "AveragePool-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnPool<T>> kernel;
        kernel = std::make_shared<MklDnnPool<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "GlobalAveragePool") {
        std::ostringstream os;
        os << "GlobalAveragePool-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnPool<T>> kernel;
        kernel = std::make_shared<MklDnnPool<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "LRN") {
        std::ostringstream os;
        os << "LRN-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnLrn<T>> kernel;
        kernel = std::make_shared<MklDnnLrn<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      } else if (mkldnn_node.name == "Sum") {
        std::ostringstream os;
        os << "Sum-" << mkldnn_node.node_index << "-";
        std::shared_ptr<MklDnnSum<T>> kernel;
        kernel = std::make_shared<MklDnnSum<T>>(mkldnn_node, params.provider, params.attributes, os.str());
        for (auto index : mkldnn_node.parent_nodes) {
          kernel->parents_.push_back(context_.kernels[index]);
        }
        context_.kernels.push_back(kernel);
      }
    }
  }

  struct SubgraphContext {
    std::unique_ptr<mkldnn::stream> stream;
    Vector<mkldnn::primitive> net;
    Vector<std::unordered_map<int, mkldnn::memory>> net_args;
    Vector<std::shared_ptr<MklDnnKernel>> kernels;

    SubgraphContext() : stream(nullptr) {}
  };

  void Initialize(const OrtCustomOpApi* api, OrtKernelContext* context) {
    // Propagate mkldnn block format
    // dst format of current node to src format of next node
    for (auto& kernel : context_.kernels) {
      kernel->CreatePrimitives(api, context, cpu_engine_, context_.net, context_.net_args);
      if (kernel->primitive_created_status_.IsOK()) {
        kernel->ReorderWeights(api, context, cpu_engine_);
      }
    }
  }

  SubgraphContext context_;
  mkldnn::engine& cpu_engine_;
};

// Pool which allows for reuse of MKLDNN Conv primitives which are expensive to instantiate.
// To address thread safety, the primitives are stored in a map on thread local storage.
template <typename T>
class SubgraphPrimitivePool : public PrimitivePool<T> {
 public:
  static SubgraphPrimitive<T>* Get(const OrtCustomOpApi* api,
                                   OrtKernelContext* context,
                                   const SubgraphParams& params) {
    Ort::CustomOpApi ort{*api};
    std::string dims_str;
    for (auto i = 0; i < params.subgraph->mkldnn_nodes[0].num_inputs; i++) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      auto shape = tensor_shape.data();
      auto dim = tensor_shape.size();

      TensorShape x_shape(shape, dim);
      mkldnn::memory::dims src_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());
      AddDimsToKey(dims_str, src_dims);
    }

    SubgraphPrimitive<T>* primitive = dynamic_cast<SubgraphPrimitive<T>*>(
        SubgraphPrimitivePool<T>::GetInstance().GetPrimitive(params.subgraph_key + dims_str));

    if (primitive == nullptr) {
      auto subgraph_primitive = onnxruntime::make_unique<SubgraphPrimitive<T>>(api, context, params);
      primitive = subgraph_primitive.get();
      SubgraphPrimitivePool<T>::GetInstance().SetPrimitive(params.subgraph_key + dims_str, std::move(subgraph_primitive));
    }
    return primitive;
  }

 private:
  SubgraphPrimitivePool() = default;
  ~SubgraphPrimitivePool() = default;

  static SubgraphPrimitivePool& GetInstance() {
    static SubgraphPrimitivePool pool;
    return pool;
  }
};
}  // namespace

template <typename T>
Status MkldnnFuncKernel<T>::Compute(const OrtCustomOpApi* api, OrtKernelContext* context) const {
  Status status;
  try {
    SubgraphPrimitive<T>* primitive = SubgraphPrimitivePool<T>::Get(api, context, params_);
    primitive->UpdateProvider(params_);
    status = primitive->Compute(api, context);
  } catch (const mkldnn::error& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Status: ", e.status,
                           ", message: ", e.what());
  }
  return status;
}

template class MkldnnFuncKernel<float>;

}  // namespace mkl_dnn
}  // namespace onnxruntime
