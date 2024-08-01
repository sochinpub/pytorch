import threading
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch._utils import ExceptionWrapper
from torch.cuda._utils import _get_device_index

from ..modules import Module


__all__ = ["get_a_var", "parallel_apply"]


def get_a_var(
    obj: Union[torch.Tensor, List[Any], Tuple[Any, ...], Dict[Any, Any]],
) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Optional[Sequence[Dict[str, Any]]] = None,
    devices: Optional[Sequence[Optional[Union[int, torch.device]]]] = None,
) -> List[Any]:
    r"""Apply each `module` in :attr:`modules` in parallel on each of :attr:`devices`.
        # threading 多线程实现，用前面准备好的 replica 和输入数据，然后
        # for 循环启动多线程

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    # 每个 GPU 都有模型和输入
    assert len(modules) == len(
        inputs
    ), f"The number of modules {len(modules)} is not equal to the number of inputs {len(inputs)}"
    # # 确保每个 GPU 都有相应的数据，如没有就空白补全
    if kwargs_tup is not None:
        # 在 scatter 已经补全了
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = (cast(Dict[str, Any], {}),) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    # 拿到设备和streams
    devices = [_get_device_index(x, True) for x in devices]
    streams = [torch.cuda.current_stream(x) for x in devices]
    # 多线程实现
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = (
        torch.is_grad_enabled(),
        torch.is_autocast_enabled(),
    )
    # 定义 worker
    def _worker(
        i: int,
        module: Module,
        input: Any,
        kwargs: Dict[str, Any],
        device: Optional[Union[int, torch.device]] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            t = get_a_var(input)
            if t is None:
                with lock:
                    results[i] = ExceptionWrapper(
                        where=f"in replica {i}, no device was provided and no tensor input was found; "
                        "device cannot be resolved"
                    )
                return
            device = t.get_device()
        if stream is None:
            stream = torch.cuda.current_stream(device)
        try:
            with torch.cuda.device(device), torch.cuda.stream(
                stream
            ), torch.amp.autocast("cuda", enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock: # 并行计算得到输出
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where=f"in replica {i} on device {device}"
                )
    # 如有一个进程控制多个 GPU ，起多个线程
    # 需要强调一下，虽然 DDP 推荐单卡单进程，即每次调用 DDP device_ids 都只输入一张卡的 id（通常是 args.local_rank）
    # 但是如果输入多个 device_id，此时 DDP 就是单进程多线程控制多卡，和 DP 一样，关于 DDP 的解读可以看下文
    if len(modules) > 1:
        threads = [
            threading.Thread(
                target=_worker, args=(i, module, input, kwargs, device, stream)
            )
            for i, (module, input, kwargs, device, stream) in enumerate(
                zip(modules, inputs, kwargs_tup, devices, streams)
            )
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else: # 一个 GPU 一个进程 （ DDP 推荐操作）
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs
