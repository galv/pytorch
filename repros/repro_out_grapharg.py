### Mini Repo ###
import torch
from torch._higher_order_ops.while_loop import while_loop

a = torch.zeros((4,), device="cuda")
b = torch.full((4,), fill_value=0, dtype=torch.bool, device="cuda")
c = torch.ones((4,), device="cuda")
z = torch.tensor(0, device="cuda")

def f(z, a, b):
    def cond_fn(it):
        return it.sum() < 5

    def body_fn(it):
        torch.ge(it, a, out=b)
        # the following can work
        # b_ = torch.ge(it, a)
        # b.copy_(b_)
        return (it + 1)

    _ = while_loop(cond_fn, body_fn, (z,))

f(z, a, b)
