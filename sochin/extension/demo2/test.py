
from torch.autograd import Function
import torch
import time
import sum_double

class SumDouble(Function):
    @staticmethod
    def forward(ctx, array1, array2):
        array1 = array1.float()
        array2 = array2.float()
        ans = array1.new_zeros(array1.shape)
        sum_double.forward(array1.contiguous(), array2.contiguous(), ans)
        return ans
    @staticmethod
    def backward(ctx, g_out):
        g_in1 = g_out.clone()
        g_in2 = g_out.clone()
        return g_in1, g_in2
    
sum_double_op = SumDouble.apply

class Timer:
    def __init__(self, op_name):
        self.begin_time = 0
        self.end_time = 0
        self.op_name = op_name
        
    def __enter__(self):
        torch.cuda.synchronize()
        self.begin_time = time.time()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()
        print(f"Average time cost of {self.op_name} is {(self.end_time - self.begin_time) * 1000:.4f} ms")
        
if __name__ == '__main__':
    n = 1000000
    device = torch.device("cuda0" if torch.cuda.is_available() else "cpu")
    tensor1 = torch.ones(n, dtype=torch.float32, device=device, requires_grad=True)
    tensor2 = torch.ones(n, dtype=torch.float32, device=device, requires_grad=True)
    with Timer("sum_double"):
        ans = sum_double_op(tensor1, tensor2)