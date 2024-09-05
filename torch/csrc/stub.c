#include <Python.h>

extern PyObject* initModule(void);

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);
#endif
// 解释器的模块初始化函数， 初始化 _C 模块
PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}
