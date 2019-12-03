#pragma once
#ifdef USE_MIMALLOC
#include <stdio.h>
#include <mimalloc.h>
#include <cassert> // for assert
#include <limits>  // for max_size

template <class T>
struct allocator_mimalloc {
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  allocator_mimalloc() noexcept {}
  allocator_mimalloc(const allocator_mimalloc& other) noexcept {}
  template <class U>
  allocator_mimalloc(const allocator_mimalloc<U>& other) noexcept {}

  pointer address(reference x) const noexcept { return &x; }
  const_pointer address(const_reference x) const noexcept { return &x; }

  size_type max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  pointer allocate(size_type n, const void* hint = 0);
  void deallocate(T* p, std::size_t n);

  template <class U, class... Args>
  void construct(U* p, Args&&... args) { assert(false); }
  template <class U>
  void destroy(U* p) { assert(false); }
};

template <class T>
T* allocator_mimalloc<T>::allocate(size_type n, const void* hint) {
  //printf("[allocate] %zd of size %zd bytes each\n", n, sizeof(T));
  //auto x = (T*)mi_malloc(n * sizeof(T));
  auto x = (T*)mi_mallocn(n, sizeof(T));
  //printf("      [allocate] done - %p\n", x);
  return x;
}

template <class T>
void allocator_mimalloc<T>::deallocate(T* p, std::size_t n) {
  //printf("[free] %p %zd\n", p, n);
  mi_free(p);
  //printf("      [free] done\n");
}

template <class T1, class T2>
bool operator==(const allocator_mimalloc<T1>& lhs, const allocator_mimalloc<T2>& rhs) noexcept { return true; }
template <class T1, class T2>
bool operator!=(const allocator_mimalloc<T1>& lhs, const allocator_mimalloc<T2>& rhs) noexcept { return false; }

template <typename T>
using Tensor_Alloc = allocator_mimalloc<T>;

#else
template <typename T>
using Tensor_Alloc = std::allocator;
#endif

// std::vector<int, Tensor_Alloc<int> > x;