"""Regression test for batch-padding invariance in the audio position embedding."""

from __future__ import annotations

import torch

from f5_tts.model.modules import ConvPositionEmbedding


def main() -> None:
    torch.manual_seed(666)
    module = ConvPositionEmbedding(dim=32).eval()
    short_length = 100
    long_length = 300
    short = torch.randn(1, short_length, 32)
    long = torch.randn(1, long_length, 32)

    short_mask = torch.ones(1, short_length, dtype=torch.bool)
    batch_mask = torch.arange(long_length).unsqueeze(0) < torch.tensor([[short_length], [long_length]])
    padded_batch = torch.zeros(2, long_length, 32)
    padded_batch[0, :short_length] = short[0]
    padded_batch[1] = long[0]

    with torch.inference_mode():
        standalone = module(short, mask=short_mask)
        batched = module(padded_batch, mask=batch_mask)

    valid_batched = batched[0, :short_length]
    max_abs = (standalone[0] - valid_batched).abs().max().item()
    if not torch.allclose(standalone[0], valid_batched, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"Valid position embeddings depend on batch padding: max_abs={max_abs}")
    if torch.count_nonzero(batched[0, short_length:]).item() != 0:
        raise AssertionError("Position embedding repopulated padded frames")

    print(f"PADDING_SAFE_POSITION_EMBEDDING_OK max_abs={max_abs:.9g}", flush=True)


if __name__ == "__main__":
    main()
