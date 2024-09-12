
import torch
import torch.distributed as dist

class OrionAckWork(dist._Work):
    pass

class OrionAckProcessGroup(dist.ProcessGroup):
    def getBackendName(self):
        return "OrionAck"
    
    def allgather(self, output_tensor_list, input_tensor_list, opts = None):
        pass
    
    def allreduce(self, tensor_list, opts = None):
        pass
    
    def barrier(self, opts=...) -> dist.Work:
        pass

    def broadcast(self, tensor_list, opts = None):
        pass
    
    def send(self, tensor_list, dst, tag = 0):
        pass
    
    def recv(self, tensor_list, src, tag = 0):
        pass
    
