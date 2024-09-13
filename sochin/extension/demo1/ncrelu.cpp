

#include <torch/extension.h>

/**
 * 由于我们是利用Pytorch中ATen张量库封装的高层操作，是一种与运行设备无关的代码抽象，
 * 因此所实现的函数可以直接应用于GPU上进行计算，只需要将输入迁移至GPU上即可
 * ref: https://www.zhihu.com/tardis/zm/art/350651297?source_id=1003
 */

/**************************************************
 * 
 * 第一步：使用C++编写算子的forward函数和backward函数
 * 第二步：将该算子的forward函数和backward函数使用**pybind11**绑定到python上
 * 第三步：使用setuptools/JIT/CMake编译打包C++工程为so文件
 * 
 ****************************************************/
torch::Tensor ncrelu_forward(torch::Tensor input)
{
    auto positive = input.clamp_min(0);
    auto negative = input.clamp_max(0);
    return torch::cat({positive, negative}, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ncrelu_forward, "NCReLU fowward");
}