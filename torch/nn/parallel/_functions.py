import warnings
from typing import List, Optional

import torch
from torch._utils import _get_device_index
from torch.autograd import Function

from . import comm


class Broadcast(Function):
    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        assert all(
            i.device.type != "cpu" for i in inputs
        ), "Broadcast function not implemented for CPU tensors"
        # 获取 broadcast的GPU
        # inputs是设备
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.target_gpus = target_gpus
        if len(inputs) == 0:
            return ()
        ctx.num_inputs = len(inputs)
        # input 放在 device[0]
        ctx.input_device = inputs[0].get_device()
        # 和 detach 的情况一样，调用集合通信的broadcast
        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
        # comm.broadcast_coalesced 的代码
        # tensors 必须在同一个设备，CPU 或者 GPU； devices 即是要拷贝到的设备；buffer_size 则是最大的buffer
        # 这里用到 buffer 将小张量合并到缓冲区以减少同步次数
        # def broadcast_coalesced(tensors, devices, buffer_size=10485760):
        #    devices = [_get_device_index(d) for d in devices]
        #       return torch._C._broadcast_coalesced(tensors, devices, buffer_size)
        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                for output in outputs:
                    non_differentiables.append(output[idx])
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple([t for tensors in outputs for t in tensors])

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + ReduceAddCoalesced.apply(
            ctx.input_device, ctx.num_inputs, *grad_outputs
        )


class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        ctx.target_gpus = [
            grads[i].get_device() for i in range(0, len(grads), num_inputs)
        ]

        grads_ = [grads[i : i + num_inputs] for i in range(0, len(grads), num_inputs)]
        return comm.reduce_add_coalesced(grads_, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (
            None,
            None,
        ) + Broadcast.apply(ctx.target_gpus, *grad_outputs)


class Gather(Function):
    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(
            i.device.type != "cpu" for i in inputs
        ), "Gather function not implemented for CPU tensors"
        if target_device == "cpu":
            ctx.target_device = "cpu"
        else:
            target_device = _get_device_index(target_device, True)
            ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(i.get_device() for i in inputs)
        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn(
                "Was asked to gather along dimension 0, but all "
                "input tensors were scalars; will instead unsqueeze "
                "and return a vector."
            )
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(i.size(ctx.dim) for i in inputs)
        return comm.gather(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        scattered_grads = Scatter.apply(
            ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output
        )
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads


class Scatter(Function):
    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        # 获取GPU索引
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        if torch.cuda.is_available() and ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream
            # 在当前设备上，新建cuda stream
            streams = [
                _get_stream(torch.device("cuda", device)) for device in target_gpus
            ]
        # 真正的集合操作
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)
        # Synchronize with the copy stream
        # sync一下各个设备（stream上）的异步拷贝：等待拷贝完成
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


# background streams used for copying
_streams: Optional[List[Optional[torch.Stream]]] = None


def _get_stream(device: torch.device):
    """Get a background stream for copying between CPU and target device.
        创建一个cuda strem
    """
    global _streams
    if device.type == "cpu":
        return None
    device_mod = getattr(torch, device.type, None)
    if device_mod is None:
        return None
    if _streams is None:
        _streams = [None] * device_mod.device_count()
    if _streams[device.index] is None:
        _streams[device.index] = device_mod.Stream(device.index)
    return _streams[device.index]
