"""
This is refactored from cmake.py to avoid circular imports issue with env.py,
which calls get_cmake_cache_variables_from_file
"""

from __future__ import annotations

import re
from typing import IO, Optional, Union


CMakeValue = Optional[Union[bool, str]] # 定义cmake的值为可选参数：可以是bool，也可以str


def convert_cmake_value_to_python_value(
    cmake_value: str,                   # 值
    cmake_type: str                     # 类型
) -> CMakeValue:
    r"""Convert a CMake value in a string form to a Python value.
            转换一个字符串格式的值为python值
    Args:
      cmake_value (string): The CMake value in a string form (e.g., "ON", "OFF", "1").
      cmake_type (string): The CMake type of :attr:`cmake_value`.

    Returns:
      A Python value corresponding to :attr:`cmake_value` with type :attr:`cmake_type`.
    """

    cmake_type = cmake_type.upper()
    up_val = cmake_value.upper()
    if cmake_type == "BOOL":                    # bool 类型
        # https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/VariablesListsStrings#boolean-values-in-cmake
        return not (
            up_val in ("FALSE", "OFF", "N", "NO", "0", "", "NOTFOUND")
            or up_val.endswith("-NOTFOUND")
        )
    elif cmake_type == "FILEPATH":              # 字符串类型
        if up_val.endswith("-NOTFOUND"):
            return None
        else:
            return cmake_value
    else:  # Directly return the cmake_value.
        return cmake_value


def get_cmake_cache_variables_from_file(
    cmake_cache_file: IO[str],
) -> dict[str, CMakeValue]:
    r"""Gets values in CMakeCache.txt into a dictionary.

    Args:
      cmake_cache_file: A CMakeCache.txt file object. CMakeCache.txt文件对象
    Returns:
      dict: A ``dict`` containing the value of cached CMake variables.
    """

    results = {}
    for i, line in enumerate(cmake_cache_file, 1):      # 从文件第一行开始迭代
        line = line.strip()
        if not line or line.startswith(("#", "//")):    # 过滤注释
            # Blank or comment line, skip
            continue

        # Almost any character can be part of variable name and value. As a practical matter, we assume the type must be
        # valid if it were a C variable name. It should match the following kinds of strings:
        # 几乎任何字符都可以作为变量名和值的一部分。作为一个实际问题，我们假定类型如果是C变量名则必须是有效的
        # 则有效。它应该匹配以下类型的字符串
        #   USE_CUDA:BOOL=ON
        #   "USE_CUDA":BOOL=ON
        #   USE_CUDA=ON
        #   USE_CUDA:=ON
        #   Intel(R) MKL-DNN_SOURCE_DIR:STATIC=/path/to/pytorch/third_party/ideep/mkl-dnn
        #   "OpenMP_COMPILE_RESULT_CXX_openmp:experimental":INTERNAL=FALSE
        matched = re.match(
            r'("?)(.+?)\1(?::\s*([a-zA-Z_-][a-zA-Z0-9_-]*)?)?\s*=\s*(.*)', line
        )
        if matched is None:  # Illegal line
            raise ValueError(f"Unexpected line {i} in {repr(cmake_cache_file)}: {line}")
        _, variable, type_, value = matched.groups()                                            # 匹配拿到的变量名， 类型和值
        if type_ is None:
            type_ = ""
        if type_.upper() in ("INTERNAL", "STATIC"):
            # CMake internal variable, do not touch
            continue
        results[variable] = convert_cmake_value_to_python_value(value, type_)

    return results
