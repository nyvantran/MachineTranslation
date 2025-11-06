import torch
import functools
from torch.utils.checkpoint import create_selective_checkpoint_contexts


def fn(x, y):
    a = x ** 2
    b = y ** 2
    c = a + b
    d = torch.sin(c)
    return d


x = torch.randn(3, 3, requires_grad=True)
y = torch.randn(3, 3, requires_grad=True)


def policy_fn(ctx, op, *args, **kwargs):
    print(ctx.is_recompute)
    context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
    out = torch.utils.checkpoint.checkpoint(
        fn, x, y,
        use_reentrant=False,
        context_fn=context_fn,
    )
