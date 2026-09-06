"""Tiny four-rank gradient/unused-parameter regression; no dataset or artifacts."""
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from test_tpca_attention import inputs, tiny_model
from torch.nn.parallel import DistributedDataParallel

from aligndit.model.cfm_vt import CFM_VT


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    torch.manual_seed(666)
    model = CFM_VT(transformer=tiny_model(), mel_spec_kwargs={"n_mel_channels": 8},
                   ctc_lambda=.1, tpca_visual_ctc_lambda=.03, tpca_path_lambda=.01).cuda(rank)
    ddp = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-4)
    args = inputs()
    for key, value in args.items():
        args[key] = value.cuda(rank)
    seen_path = False
    for update in range(8):
        optimizer.zero_grad(set_to_none=True)
        model.transformer.set_tpca_step(0 if update < 2 else 4)
        # Different ranks exercise full, TTS, VTS and null branches together.
        condition_value = [0.9, 0.1, 0.3, 0.5][(update + rank) % 4]
        with patch("aligndit.model.cfm_vt.random", side_effect=[.99, condition_value]), \
                torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components, _, _ = ddp(
                args["x"], args["text"], args["video"], lens=args["mask"].sum(1),
                text_lens=args["text_mask"].sum(1), video_lens=args["video_mask"].sum(1),
            )
        assert torch.isfinite(loss)
        seen_path |= components["tpca_path_loss"] > 0
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
        optimizer.step()
    assert seen_path
    params = next(model.transformer.tpca_aligner.parameters()).detach()
    expected = params.clone()
    dist.broadcast(expected, 0)
    torch.testing.assert_close(params, expected)
    dist.barrier()
    if rank == 0:
        print("PASS: four-rank BF16 DDP, mixed CFG, warmup-to-active transition, finite synchronized gradients", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
