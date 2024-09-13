#pragma once

#include <memory>
#include <type_traits>

#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>

namespace c10 {

// Compatibility wrapper around a raw pointer so that existing code
// written to deal with a shared_ptr can keep working.
// 兼容wrapper 一个指针
template <typename T>
class SingletonTypePtr {
 public:
  /* implicit */ SingletonTypePtr(T* p) : repr_(p) {}

  // We need this to satisfy Pybind11, but it shouldn't be hit.
  // 这个显示的构造只为了满足Pybind11，但不应该被调用
  explicit SingletonTypePtr(std::shared_ptr<T>) { TORCH_CHECK(false); } 

  using element_type = typename std::shared_ptr<T>::element_type;

  // 解引用
  template <typename U = T, std::enable_if_t<!std::is_same_v<std::remove_const_t<U>, void>, bool> = true>
  T& operator*() const {
    return *repr_;
  }
  // .get
  T* get() const {
    return repr_;
  }
  // ->
  T* operator->() const {
    return repr_;
  }
  // 转bool类型： operator Typename()
  operator bool() const {
    return repr_ != nullptr;
  }

 private:
  T* repr_{nullptr};
};
// 模板函数：判断包装了相同的指针
template <typename T, typename U>
bool operator==(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {
  return (void*)lhs.get() == (void*)rhs.get();
}

template <typename T, typename U>
bool operator!=(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {
  return !(lhs == rhs);
}

} // namespace c10
