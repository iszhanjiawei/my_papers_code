"""Compare original dense-mask and split-query SDPA; no training data required.

Run on an idle CUDA GPU for timing. This isolates attention, not full training.
"""
import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(666)
    torch.set_num_threads(1)
    device = 'cuda'
    for batch, length in ((8, 400), (4, 800)):
        shape = (batch, 12, 2 * length, 64)
        tensors = [torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        local = (torch.arange(length, device=device)[:, None] - torch.arange(length, device=device)[None]).abs() <= 2
        dense_mask = torch.ones((1, 1, 2 * length, 2 * length), device=device, dtype=torch.bool)
        dense_mask[:, :, :length, length:] = local
        allowed = F.pad(local, (length, 0), value=True)[None, None]
        bias = torch.zeros_like(allowed, dtype=torch.bfloat16).masked_fill(~allowed, float('-inf'))
        def forward(split):
            q, k, v = tensors
            if split:
                return torch.cat([
                    F.scaled_dot_product_attention(q[:, :, :length], k, v, attn_mask=bias),
                    F.scaled_dot_product_attention(q[:, :, length:], k, v),
                ], dim=2)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)
        weight = torch.randn(shape, device=device, dtype=torch.bfloat16)
        outputs, gradients = [], []
        for split in (False, True):
            out = forward(split)
            outputs.append(out.detach().float())
            gradients.append([g.float() for g in torch.autograd.grad((out * weight).sum(), tensors)])
        def errors(a, b):
            delta = a - b
            return dict(max_abs=delta.abs().max().item(), relative_rms=(delta.square().mean().sqrt() / a.square().mean().sqrt()).item())
        result = dict(batch=batch, tokens_per_stream=length, output=errors(*outputs), gradients=[errors(a, b) for a, b in zip(*gradients)])
        # bf16 kernels need not agree bitwise; bound aggregate numerical error.
        assert result['output']['relative_rms'] < 0.01
        assert all(g['relative_rms'] < 0.02 for g in result['gradients'])
        for split in (False, True):
            name = 'split' if split else 'dense'
            def step():
                for tensor in tensors:
                    tensor.grad = None
                (forward(split) * weight).sum().backward()
            for _ in range(3):
                step()
            times = []
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                step()
                torch.cuda.synchronize()
                times.append(1000 * (time.perf_counter() - start))
            result[name + '_forward_backward_ms'] = statistics.median(times)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                step()
                torch.cuda.synchronize()
            result[name + '_operators'] = [entry.key for entry in prof.key_averages() if 'scaled_dot_product' in entry.key]
        result['attention_speedup'] = result['dense_forward_backward_ms'] / result['split_forward_backward_ms']
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
