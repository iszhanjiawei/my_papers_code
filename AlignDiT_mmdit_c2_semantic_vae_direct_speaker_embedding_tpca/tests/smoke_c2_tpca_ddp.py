"""Eight synthetic BF16 DDP regression iterations; no data or saved artifacts.

Run with ``PYTHONPATH=src torchrun --standalone --nproc_per_node=4
tests/smoke_c2_tpca_ddp.py``. This exercises mixed condition-drop branches
across ranks, the TPCA warmup transition and speaker gradient reduction.
"""

import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from aligndit.model.cfm_vt import CFM_VT
from test_c2_tpca_integration import FakeLatentAdapter, inputs, open_gates, tiny_model


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    torch.manual_seed(666)
    backbone = tiny_model()
    open_gates(backbone)
    model = CFM_VT(
        transformer=backbone, mel_spec_module=FakeLatentAdapter(),
        audio_video_ratio=1, ctc_lambda=0.03,
        tpca_visual_ctc_lambda=0.03, tpca_path_lambda=0.01,
    ).cuda(rank)
    ddp = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-4)
    torch.manual_seed(667 + rank)
    args = {name: value.cuda(rank) for name, value in inputs().items()}
    seen_path = False
    for update in range(8):
        optimizer.zero_grad(set_to_none=True)
        model.transformer.set_tpca_step(0 if update < 2 else 4)
        model.ctc_lambda = 0.0 if update < 2 else 0.03
        condition_value = [0.9, 0.1, 0.3, 0.5][(update + rank) % 4]
        with patch("aligndit.model.cfm_vt.random", side_effect=[0.99, condition_value]), \
                torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components, _, _ = ddp(
                args["x"], args["text"], args["video"],
                speaker_embedding=args["speaker_embedding"],
                lens=args["mask"].sum(1), text_lens=args["text_mask"].sum(1),
                video_lens=args["video_mask"].sum(1),
            )
        assert torch.isfinite(loss)
        seen_path |= components["tpca_path_loss"] > 0
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
        for parameter in (model.transformer.speaker_proj.weight,
                          model.transformer.tpca_aligner.output_proj.weight):
            assert parameter.grad is not None
            assert parameter.grad.abs().sum() > 0
        optimizer.step()
    assert seen_path
    for parameter in (model.transformer.speaker_proj.weight,
                      model.transformer.tpca_aligner.output_proj.weight):
        reference = parameter.detach().clone()
        dist.broadcast(reference, 0)
        torch.testing.assert_close(parameter, reference)
    dist.barrier()
    if rank == 0:
        print("PASS: four-rank BF16 DDP, 8 mixed-CFG iterations, warmup-to-active, "
              "finite synchronized TPCA and speaker gradients", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
