#include <ATen/core/functional.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/Stream.h>
#include <torch/csrc/cuda/THCP.h>
#include <torch/csrc/cuda/comm.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/ATen.h>

#include <cstddef>
#include <vector>

#include <torch/csrc/profiler/unwind/unwind.h>

namespace torch::cuda::python {
// cuda集合通信的函数，c转python: 通过pybind11实现
void initCommMethods(PyObject* module) {                                        // 模块对象
  auto m = py::cast<py::module>(module);
  // def生成代码： 暴露C函数的代码给Python
  m.def(
       "_broadcast_coalesced",                                                  // 函数名
       [](std::vector<at::Tensor>& tensors,
          const std::vector<int64_t>& devices,
          size_t buffer_size) {
         return broadcast_coalesced(tensors, devices, buffer_size);
       },                                                                       // lambda 函数：指定了函数参数类型
       py::arg("tensors"),                                                      // 多个函数参数， 给定参数名
       py::arg("devices"),
       py::arg("buffer_size"),
       py::call_guard<py::gil_scoped_release>())
      .def(
          "_broadcast",
          [](at::Tensor& tensor, std::vector<int64_t> devices) {
            return broadcast(tensor, devices);
          },
          py::call_guard<py::gil_scoped_release>(),
          py::arg("tensor"),
          py::arg("devices"))
      .def(
          "_broadcast_out",
          [](at::Tensor& tensor, std::vector<at::Tensor>& out_tensors) {
            return broadcast_out(tensor, out_tensors);
          },
          py::call_guard<py::gil_scoped_release>(),
          py::arg("tensor"),
          py::arg("out"))
      .def(
          "_scatter",
          [](at::Tensor& tensor,
             std::vector<int64_t>& devices,
             std::optional<std::vector<int64_t>> chunk_sizes,
             int64_t dim,
             std::optional<py::object> py_streams) {
            std::optional<std::vector<std::optional<at::cuda::CUDAStream>>>
                streams;
            if (py_streams) { // 如果指定了cuda strem
              py::handle handle = *py_streams;
              streams = THPUtils_PySequence_to_CUDAStreamList(handle.ptr()); // 通过stream句柄，查找cuda stream
            }
            // Note: We're holding the GIL up to here. 仍然拿着全局解释器锁，这里会导致hang, 如果scatter函数不返回
            pybind11::gil_scoped_release no_gil;
            return scatter(tensor, devices, chunk_sizes, dim, streams); // comm.cpp
          },
          py::arg("tensor"),
          py::arg("devices"),
          py::arg("chunk_sizes"),
          py::arg("dim"),
          py::arg("streams"))
      .def(
          "_scatter_out",
          [](at::Tensor& tensor,
             std::vector<at::Tensor>& out_tensors,
             int64_t dim,
             std::optional<py::object> py_streams) {
            std::optional<std::vector<std::optional<at::cuda::CUDAStream>>>
                streams;
            if (py_streams) {
              py::handle handle = *py_streams;
              streams = THPUtils_PySequence_to_CUDAStreamList(handle.ptr());
            }
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return scatter_out(tensor, out_tensors, dim, streams);
          },
          py::arg("tensor"),
          py::arg("out"),
          py::arg("dim"),
          py::arg("streams"))
      .def(
          "_gather",
          [](std::vector<at::Tensor>& tensors,
             int64_t dim,
             std::optional<int32_t> destination_index) {
            return gather(tensors, dim, destination_index);
          },
          py::arg("tensors"),
          py::arg("dim"),
          py::arg("destination_index"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_gather_out",
          [](std::vector<at::Tensor>& tensors,
             at::Tensor& out_tensor,
             int64_t dim) { return gather_out(tensors, out_tensor, dim); },
          py::arg("tensors"),
          py::arg("out"),
          py::arg("dim"),
          py::call_guard<py::gil_scoped_release>());
}
} // namespace torch::cuda::python
