"""CPU integration checks for the actual sampler used by the GRID CFG scan.

A deterministic transformer gives distinct two-channel branch predictions. One
Euler step then exposes the sampler's actual guidance coefficients while also
checking that ablations retain the prompt, speaker and sampling geometry.
No checkpoint, dataset, decoder or GPU is needed.
"""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

from aligndit.model.cfm_vt import CFM_VT


_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts/grid_cfg_ablation.py"
_SPEC = importlib.util.spec_from_file_location("grid_cfg_ablation_under_test", _RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RUNNER)


class BranchProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.dim = 2
        self.calls = []
        self.cache_clears = 0

    def clear_cache(self):
        self.cache_clears += 1

    def forward(self, **kwargs):
        self.calls.append({
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        })
        # Known conditional predictions, deliberately non-collinear.
        full, text_only, video_only, null = (10., 1.), (4., 2.), (7., 5.), (1., 4.)
        active = video_only if kwargs.get("drop_text") else text_only if kwargs.get("drop_video") else full
        if not kwargs.get("cfg_infer"):
            branches = [active]
        elif kwargs.get("drop_text") or kwargs.get("drop_video"):
            branches = [active, null]
        else:
            branches = [full, text_only, null]
        template = kwargs["x"]
        predictions = [template.new_tensor(branch).expand_as(template) for branch in branches]
        return torch.cat(predictions, dim=0), {}


class GridCfgSamplingTests(unittest.TestCase):
    def test_selection_is_balanced_and_independent_of_input_order_or_scores(self):
        rows = []
        for speaker in range(1, 34):
            for utterance in range(8):
                speaker_id, utterance_id = f"s{speaker}", f"clip{utterance}"
                rows.append({
                    "speaker": speaker_id,
                    "utterance_key": f"grid/{speaker_id}/{utterance_id}",
                    "audio_relative_path": f"val/{speaker_id}/{utterance_id}.wav",
                })
        selected = RUNNER.select_rows(rows)
        changed = [{**row, "wer": index / len(rows), "asr_hypothesis": "ignored"}
                   for index, row in enumerate(reversed(rows))]
        reordered = RUNNER.select_rows(changed)
        self.assertEqual(len(selected["ids"]), 132)
        self.assertEqual(set(selected["speaker_counts"].values()), {4})
        self.assertEqual(set(selected["ids"]), set(reordered["ids"]))
        self.assertEqual(selected["selection_scores"], reordered["selection_scores"])

    def test_eight_variants_share_noise_prompt_speaker_and_geometry(self):
        saved_rng = torch.get_rng_state()
        saved_threads = torch.get_num_threads()
        self.addCleanup(torch.set_rng_state, saved_rng)
        self.addCleanup(torch.set_num_threads, saved_threads)
        torch.set_num_threads(2)

        probe = BranchProbe()
        sampler = CFM_VT(
            transformer=probe, num_channels=2, audio_video_ratio=1,
            vocab_char_map={"a": 0}, ctc_lambda=0.,
        )
        prompt = torch.tensor([[[.1, .2], [.3, .4], [.5, .6]]])
        video = torch.arange(12).reshape(1, 6, 2).float()
        speaker = torch.ones(1, 192)
        # Expected integrated velocities are independent numeric test oracles.
        expected_velocities = {
            "cfg_t5_v2": (37., -11.),
            "cfg_t3_v2": (31., -7.),
            "cfg_t1_v2": (25., -3.),
            "cfg_t5_v1": (31., -10.),
            "cfg_t5_v0": (25., -9.),
            "cfg_t0_v0": (10., 1.),
            "drop_text_t0_v0": (7., 5.),
            "drop_video_t0_v0": (4., 2.),
        }
        self.assertEqual(set(RUNNER.VARIANTS), set(expected_velocities))
        initial_noise = None
        for index, (name, variant) in enumerate(RUNNER.VARIANTS.items(), 1):
            with self.subTest(variant=name):
                cfg_text, cfg_video = variant["cfg_text"], variant["cfg_video"]
                ignored, velocity = variant["ignore_modality"], expected_velocities[name]
                probe.calls.clear()
                output, _ = sampler.sample(
                    cond=prompt.clone(), text=["a"], duration=torch.tensor([6]),
                    video=video.clone(), lens=torch.tensor([3]), steps=1,
                    cfg_strength=cfg_text, cfg_strength_v=cfg_video,
                    seed=0, use_epss=False, ignore_modality=ignored,
                    speaker_embedding=speaker.clone(),
                )
                self.assertEqual(len(probe.calls), 1)
                call = probe.calls[0]
                if initial_noise is None:
                    initial_noise = call["x"].clone()
                self.assertTrue(torch.equal(call["x"], initial_noise))
                self.assertEqual(tuple(output.shape), (1, 6, 2))
                self.assertTrue(torch.equal(output[:, :3], prompt))
                expected_target = initial_noise[:, 3:] + initial_noise.new_tensor(velocity)
                torch.testing.assert_close(output[:, 3:], expected_target, rtol=0, atol=1e-6)
                self.assertTrue(torch.equal(call["cond"][:, :3], prompt))
                self.assertEqual(torch.count_nonzero(call["cond"][:, 3:]).item(), 0)
                self.assertTrue(torch.equal(call["video"], video))
                self.assertEqual(tuple(call["text"].shape), (1, 1))
                self.assertTrue(torch.equal(call["speaker_embedding"], speaker))
                self.assertIs(call["drop_speaker"], False)
                self.assertEqual(call["generation_mask"].tolist(), [[False] * 3 + [True] * 3])
                self.assertEqual(call["drop_text"], ignored == "text")
                self.assertEqual(call["drop_video"], ignored == "video")
                # Zero CFG is conditional sampling, not the all-null branch.
                self.assertEqual(bool(call.get("cfg_infer")), cfg_text != 0 or cfg_video != 0)
                self.assertEqual(probe.cache_clears, index)


if __name__ == "__main__":
    unittest.main()
