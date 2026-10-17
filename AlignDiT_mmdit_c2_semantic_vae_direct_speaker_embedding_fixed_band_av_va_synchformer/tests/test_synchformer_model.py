"""CPU regressions for temporal alignment, CFG isolation and checkpoint safety."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.synchformer_conditioning import resample_sync


torch.set_num_threads(1)


def model_kwargs(**overrides):
    args = dict(dim=32, depth=3, heads=2, dim_head=16, mel_dim=4, text_dim=16,
                video_dim=6, text_num_embeds=12, n_mm_layers=1, n_text_layers=2,
                conv_layers=0, use_conformer=False, layer_indices_ctc=[],
                audio_video_ratio=1, speaker_dim=5, dropout=0, attn_mask_enabled=True,
                temporal_band_enabled=True, temporal_band_mode='fixed',
                temporal_band_bidirectional=True)
    return {**args, **overrides}


def inputs():
    mask = torch.arange(12)[None] < torch.tensor([10, 12])[:, None]
    generation = mask.clone()
    generation[:, :3] = False
    return dict(x=torch.randn(2,12,4), cond=torch.randn(2,12,4), video=torch.randn(2,12,6),
                text=torch.tensor([[1,2],[3,4]]), time=torch.tensor([.2,.7]), mask=mask,
                text_mask=torch.ones(2,2,dtype=torch.bool), video_mask=mask,
                complementary_mask=~generation & mask, generation_mask=generation,
                sync_feat=torch.randn(2,16,8), sync_lens=torch.tensor([8,16]),
                speaker_embedding=torch.randn(2,5))


def trained_model(**kwargs):
    torch.manual_seed(14)
    model = DiT_VT_MMDiT(**model_kwargs(sync_dim=8, **kwargs)).eval()
    # Mimic nonzero pretrained AdaLN/output weights: zero initialized outputs
    # alone would make all CFG and sync comparisons vacuous.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(key in name for key in ('attn_norm.linear', 'cross_attn_ada',
                                           'norm_out.linear', 'proj_out', 'speaker_proj',
                                           'sync_in.2.w2')):
                parameter.normal_(std=.07)
    return model


class SynchformerModelTests(unittest.TestCase):
    def test_nearest_exact_variable_lengths_shift_and_truncation(self):
        source = torch.arange(32.).reshape(2,16,1)
        actual = resample_sync(source, torch.tensor([8,16]), torch.tensor([9,13]), 12,
                               start=torch.tensor([2,4]))
        for b, (src_len, target_len, start) in enumerate([(8,9,2), (16,13,4)]):
            ref = torch.nn.functional.interpolate(source[b,:src_len].T[None],
                                                  size=target_len, mode='nearest-exact')[0].T
            expected = torch.zeros(12,1)
            expected[start:min(12,start+target_len)] = ref[:min(target_len,12-start)]
            torch.testing.assert_close(actual[b], expected, rtol=0, atol=0)

    def test_padding_and_prompt_are_excluded_before_modulation(self):
        model = trained_model()
        args = inputs()
        t = model.time_embed(args['time'])
        def delta(features):
            return model.get_sync_delta(features, args['sync_lens'], t, audio_len=12,
                                        generation_mask=args['generation_mask'], audio_mask=args['mask'])
        expected = delta(args['sync_feat'])
        poisoned = args['sync_feat'].clone()
        poisoned[0,8:] = float('nan')
        actual = delta(poisoned)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual[~args['generation_mask']].count_nonzero().item(), 0)
        # Valid output for a short clip is independent of another item's padding.
        short = model.get_sync_delta(args['sync_feat'][:1,:8], torch.tensor([8]), t[:1],
                                    audio_len=10, generation_mask=args['generation_mask'][:1,:10])
        torch.testing.assert_close(short, expected[:1,:10])
        actual.sum().backward()
        self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in model.sync_in.parameters()))

    def test_packed_cfg_matches_three_separate_branches_for_batch_two(self):
        model, args = trained_model(), inputs()
        packed = model(**args, cfg_infer=True)[0]
        separate = torch.cat([model(**args)[0], model(**args, drop_video=True)[0],
                              model(**args, drop_video=True, drop_text=True, drop_audio_cond=True)[0]])
        torch.testing.assert_close(packed, separate, rtol=2e-5, atol=2e-5)
        changed = {**args, 'sync_feat': args['sync_feat'] + 5}
        changed_packed = model(**changed, cfg_infer=True)[0]
        self.assertGreater((packed[:2]-changed_packed[:2]).abs().max().item(), 1e-5)
        torch.testing.assert_close(packed[2:], changed_packed[2:], rtol=0, atol=0)
        for drop in ('drop_text', 'drop_video'):
            packed_two = model(**args, **{drop: True}, cfg_infer=True)[0]
            separate_two = torch.cat([model(**args, **{drop: True})[0], separate[4:]])
            torch.testing.assert_close(packed_two, separate_two, rtol=2e-5, atol=2e-5)

    def test_video_modulation_remains_global_timestep_and_audio_is_temporal(self):
        model, args = trained_model(), inputs()
        seen = []
        block = model.transformer_blocks[0]
        handle = block.v_attn_norm.register_forward_pre_hook(
            lambda module, positional, kw: seen.append(kw['emb'].detach().clone()), with_kwargs=True)
        audio_seen = []
        handle2 = block.register_forward_pre_hook(lambda module, positional: audio_seen.append(positional[2]))
        model(**args)
        handle.remove()
        handle2.remove()
        torch.testing.assert_close(seen[0], model.time_embed(args['time']))
        self.assertEqual(audio_seen[0].shape, (2,12,32))
        torch.testing.assert_close(audio_seen[0][:,:3], seen[0][:,None].expand(-1,3,-1))

    def test_zero_projection_preserves_parent_and_receives_gradients(self):
        parent_path = Path(__file__).resolve().parents[2] / 'AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_fixed_band_av_va/src/aligndit/model/backbone/dit_vt_mm.py'
        spec = importlib.util.spec_from_file_location('synchformer_test_parent', parent_path)
        parent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parent)
        torch.manual_seed(42)
        original = parent.DiT_VT_MMDiT(**model_kwargs()).eval()
        torch.manual_seed(42)
        disabled = DiT_VT_MMDiT(**model_kwargs()).eval()
        self.assertEqual(set(original.state_dict()), set(disabled.state_dict()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
        # Use a nontrivial loaded backbone to check numerics and trainability.
        trained = trained_model()
        common = {k:v for k,v in trained.state_dict().items() if not k.startswith('sync_')}
        original.load_state_dict(common)
        disabled.load_state_dict(common)
        enabled = DiT_VT_MMDiT(**model_kwargs(sync_dim=8)).eval()
        missing = enabled.load_state_dict(common, strict=False).missing_keys
        self.assertEqual(len(missing), 6)
        args = inputs()
        base_args = {k:v for k,v in args.items() if not k.startswith('sync_')}
        expected = original(**base_args)[0]
        torch.testing.assert_close(disabled(**base_args)[0], expected, rtol=0, atol=0)
        actual = enabled(**args)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        grad = enabled.sync_in[2].w2.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_checkpointed_backward_matches_regular(self):
        model = trained_model().train()
        checkpointed = trained_model(checkpoint_activations=True).train()
        args = inputs()
        y = model(**args)[0]
        cy = checkpointed(**args)[0]
        torch.testing.assert_close(y, cy)
        y.square().mean().backward()
        cy.square().mean().backward()
        torch.testing.assert_close(model.sync_in[2].w2.weight.grad,
                                   checkpointed.sync_in[2].w2.weight.grad)

    def test_cfm_training_and_sample_pass_sync_target_alignment(self):
        model = trained_model()
        cfm = CFM_VT(transformer=model, num_channels=4, audio_video_ratio=1,
                     ctc_lambda=0, audio_drop_prob=0, cond_drop_prob=0,
                     text_drop_prob=0, video_drop_prob=0, frac_lengths_mask=(.5,.5),
                     odeint_kwargs={'method': 'euler'})
        args = inputs()
        loss, _, _, pred = cfm(inp=args['x'], text=args['text'], video=args['video'],
                              speaker_embedding=args['speaker_embedding'],
                              sync_feat=args['sync_feat'], sync_lens=args['sync_lens'],
                              lens=torch.tensor([10,12]), video_lens=torch.tensor([10,12]))
        self.assertTrue(loss.isfinite())
        loss.backward()
        captured = []
        handle = model.register_forward_pre_hook(
            lambda module, positional, kw: captured.append(kw), with_kwargs=True)
        out, _ = cfm.sample(cond=args['cond'][:,:3], text=args['text'], video=args['video'],
                            speaker_embedding=args['speaker_embedding'], duration=torch.tensor([12,12]),
                            lens=torch.tensor([2,3]), sync_feat=args['sync_feat'], sync_lens=args['sync_lens'],
                            sync_target_lens=torch.tensor([10,13]), max_duration=10,
                            steps=1, use_epss=False, cfg_strength=0, cfg_strength_v=0)
        handle.remove()
        self.assertEqual(out.shape, (2,10,4))
        torch.testing.assert_close(captured[0]['sync_start'], torch.tensor([2,3]))
        torch.testing.assert_close(captured[0]['sync_target_lens'], torch.tensor([10,13]))


if __name__ == '__main__':
    unittest.main()
