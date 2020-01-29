// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_MIMALLOC_STL_ALLOCATOR)
#include <mimalloc.h>
#else
#include "core/framework/bfc_arena.h"
#include "core/framework/allocatormgr.h"
#endif

namespace onnxruntime {

#pragma warning(disable: 4100)
template <class T>
struct bfc_allocator {
  AllocatorPtr arena;
  typedef T value_type;

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  using is_always_equal = std::true_type;

  bfc_allocator() noexcept {
    DeviceAllocatorRegistrationInfo device_info{OrtMemTypeDefault,
                                                [](int) { return onnxruntime::make_unique<TAllocator>(); },
                                                std::numeric_limits<size_t>::max()};
    arena = CreateAllocator(device_info);
  }
  bfc_allocator(const bfc_allocator& other) noexcept {
    arena = other->arena;
  }
  template <class U>
  bfc_allocator(const bfc_allocator<U>& other) noexcept {
    arena = other->arena;
  }

  T* allocate(size_t n, const void* hint = 0) {
    return (T*)arena->Alloc(n * sizeof(T));
  }

  void deallocate(T* p, size_t n) {
    arena->Free(p);
  }
};

template <class T1, class T2>
bool operator==(const bfc_allocator<T1>& lhs, const bfc_allocator<T2>& rhs) noexcept { return true; }
template <class T1, class T2>
bool operator!=(const bfc_allocator<T1>& lhs, const bfc_allocator<T2>& rhs) noexcept { return false; }

#if defined(USE_MIMALLOC_STL_ALLOCATOR)
template <typename T>
using FastAllocVector = std::vector<T,mi_stl_allocator<T>>;
#else
template <typename T>
using FastAllocVector = std::vector<T,bfc_allocator<T>>;
#endif 

}  // namespace onnxruntime
