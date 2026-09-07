"""TPCA over selected, actual AV and audio-text attention heads.

The AV conditional distribution is reconstructed from the *same* normalized,
rotated Q/K used by joint SDPA. Only its valid visual-key subdistribution is
renormalized. No auxiliary Q/K network or fixed diagonal replaces that path.
"""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def compose_path_prior(audio_query, video_key, posterior, video_mask, text_mask, smoothing):
    """Return detached R_AV @ P_VT, with last column the blank/null route."""
    logits = torch.matmul(audio_query.float(), video_key.float().transpose(-1, -2))
    logits = logits / math.sqrt(audio_query.shape[-1])
    logits = logits.masked_fill(~video_mask[:, None, None, :], -torch.inf)
    route = logits.softmax(dim=-1).nan_to_num(0.0)
    prior = torch.matmul(route, posterior[:, None].float())
    valid = torch.cat((text_mask, torch.ones_like(text_mask[:, :1])), dim=-1)
    valid = valid[:, None, None, :]
    prior = prior.masked_fill(~valid, 0.0)
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    uniform = valid.float() / valid.sum(dim=-1, keepdim=True).clamp_min(1)
    # An empty visual condition is disabled by query_mask downstream; keeping
    # a normalized fallback here also prevents NaNs on padded CFG branches.
    prior = torch.where(prior.sum(dim=-1, keepdim=True) > 0, prior, uniform)
    return (1.0 - smoothing) * prior + smoothing * uniform


def apply_tpca_audio_attention(query, key, value, baseline_output, context):
    """Soft prior + KL(raw attention), preserving unselected heads/queries.

    Inputs are B,H,T,D; context carries detached joint AV Q/K and occurrence
    posterior. Recomputing only selected heads in query chunks bounds the
    temporary R_AV matrix. The null key/value is zero and parameter-free.
    """
    heads = context["local_heads"]
    chunk_size = context["query_chunk_size"]
    active = context["query_mask"]
    bias_strength = context["bias_strength"]
    if heads <= 0 or bias_strength <= 0 or not bool(active.any()):
        return baseline_output, query.sum() * 0.0
    local_key, local_value = key[:, :heads], value[:, :heads]
    local_outputs = []
    kl_sum = query.new_zeros((), dtype=torch.float32)
    valid_keys = torch.cat(
        (context["text_mask"], torch.ones_like(context["text_mask"][:, :1])), dim=-1
    )[:, None, None, :]
    for start in range(0, query.shape[2], chunk_size):
        end = min(start + chunk_size, query.shape[2])
        q = query[:, :heads, start:end]
        with torch.autocast(device_type=query.device.type, enabled=False):
            prior = compose_path_prior(
                context["audio_query"][:, :heads, start:end],
                context["video_key"][:, :heads],
                context["posterior"],
                context["video_mask"],
                context["text_mask"],
                context["prior_smoothing"],
            )
            logits = torch.matmul(q.float(), local_key.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
            logits = torch.cat((logits, torch.zeros_like(logits[..., :1])), dim=-1)
            logits = logits.masked_fill(~valid_keys, -torch.inf)
            raw_log_probs = logits.log_softmax(dim=-1)
            log_prior = prior.clamp_min(1e-12).log()
            # Invalid keys have zero teacher mass; avoid 0 * inf in the KL.
            kl = (prior * (log_prior - raw_log_probs.masked_fill(~valid_keys, 0.0))).sum(dim=-1)
            query_mask = active[:, None, start:end]
            kl_sum = kl_sum + kl.masked_fill(~query_mask, 0.0).sum()
            probabilities = (logits + bias_strength * log_prior).softmax(dim=-1)
            # C2's original text MHA applies dropout to attention probabilities.
            # Retain that regularization on the replaced heads during training;
            # the detached teacher and the raw-attention KL remain pre-dropout.
            probabilities = F.dropout(probabilities, p=context.get("dropout_p", 0.0), training=True)
            local = torch.matmul(probabilities[..., :-1], local_value.float()).to(baseline_output.dtype)
        local_outputs.append(torch.where(
            query_mask[..., None], local, baseline_output[:, :heads, start:end]
        ))
    local_output = torch.cat(local_outputs, dim=2)
    output = torch.cat((local_output, baseline_output[:, heads:]), dim=1)
    denominator = (active.sum() * heads).clamp_min(1)
    return output, kl_sum / denominator


def apply_tpca_multihead_attention(attention, audio, text, text_mask, context):
    """Apply TPCA through C2's existing audio-only text MultiheadAttention.

    All learned Q/K/V, their biases, and the output projection are the original
    MHA parameters. The baseline heads use its SDPA dropout and padding mask;
    only selected generated-frame heads receive the occurrence prior. Text is
    never introduced into the visual stream. No checkpoint keys are added.
    """
    if not attention.batch_first or attention.bias_k is not None or attention.bias_v is not None:
        raise ValueError("TPCA requires C2 batch-first MHA without extra learned keys")
    if attention.add_zero_attn:
        raise ValueError("TPCA supplies its own parameter-free null route")
    heads = attention.num_heads
    head_dim = attention.embed_dim // heads
    batch, queries = audio.shape[:2]
    if attention.in_proj_weight is not None:
        q_weight, k_weight, v_weight = attention.in_proj_weight.chunk(3)
    else:
        q_weight, k_weight, v_weight = attention.q_proj_weight, attention.k_proj_weight, attention.v_proj_weight
    biases = (None, None, None) if attention.in_proj_bias is None else attention.in_proj_bias.chunk(3)
    query = F.linear(audio, q_weight, biases[0]).view(batch, queries, heads, head_dim).transpose(1, 2)
    key = F.linear(text, k_weight, biases[1]).view(batch, text.shape[1], heads, head_dim).transpose(1, 2)
    value = F.linear(text, v_weight, biases[2]).view(batch, text.shape[1], heads, head_dim).transpose(1, 2)
    mask = None if text_mask is None else text_mask[:, None, None, :text.shape[1]]
    dropout_p = attention.dropout if attention.training else 0.0
    baseline = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout_p)
    output, path_loss = apply_tpca_audio_attention(
        query, key, value, baseline, {**context, "dropout_p": dropout_p}
    )
    output = output.transpose(1, 2).reshape(batch, queries, attention.embed_dim)
    return F.linear(output, attention.out_proj.weight, attention.out_proj.bias), path_loss
