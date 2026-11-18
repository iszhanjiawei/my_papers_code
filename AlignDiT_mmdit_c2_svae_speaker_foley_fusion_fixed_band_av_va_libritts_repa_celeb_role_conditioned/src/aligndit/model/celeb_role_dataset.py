"""CelebV-Dub dual-transcript training from a frozen, train-only reference map."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from aligndit.model.semantic_vae_dataset import (
    SemanticVaeCelebVDubDataset,
    _read_json,
    _read_jsonl,
    sha256_file,
)


class PairedCelebVDubDataset(Dataset):
    """B supplies content and labels. A supplies the independent voice prompt."""

    def __init__(self, base_dataset: SemanticVaeCelebVDubDataset, pairing_path: str | Path):
        self.base_dataset = base_dataset
        if base_dataset.speaker_embedding_contract is None:
            raise ValueError("Role training requires the CAM++ cache used for pairing")
        path = Path(pairing_path)
        excluded_path = path.with_name("exclusions.jsonl")
        summary_path = path.with_name("summary.json")
        summary = _read_json(summary_path, label="CelebV-Dub pairing summary")
        for field, artifact in (
            ("train_manifest_sha256", base_dataset.manifest_path),
            ("pairs_sha256", path),
            ("exclusions_sha256", excluded_path),
        ):
            if summary[field] != sha256_file(artifact):
                raise ValueError(f"Pairing {field} differs from {artifact}")
        if summary["provenance"]["cam_metadata_sha256"] != base_dataset.speaker_embedding_contract["metadata_sha256"]:
            raise ValueError("Pairing and training use different CAM++ cache metadata")

        records = base_dataset.records
        by_key = {row["utterance_key"]: index for index, row in enumerate(records)}
        seen_targets = set()
        self.indices = []
        self.ctc_feasible_count = 0
        min_cosine = float(summary["settings"]["min_cosine"])
        if not math.isfinite(min_cosine) or not -1 <= min_cosine <= 1:
            raise ValueError("Invalid CAM++ cosine threshold in the pairing summary")
        for pair in _read_jsonl(path):
            target, reference = pair["target_key"], pair["reference_key"]
            if target not in by_key or reference not in by_key or target == reference or target in seen_targets:
                raise ValueError(f"Invalid or duplicate train pair: {target} <- {reference}")
            t, r = records[by_key[target]], records[by_key[reference]]
            if (
                t["split"] != "train"
                or r["split"] != "train"
                or t["video_id"] != r["video_id"]
                or target.rsplit("_", 1)[0] != reference.rsplit("_", 1)[0]
            ):
                raise ValueError(f"Pair crosses its training source or clip prefix: {target} <- {reference}")
            if not min_cosine <= pair["cosine"] <= 1.000001:
                raise ValueError(f"Pair score outside configured range: {target} <- {reference}")
            self.indices.append((by_key[target], by_key[reference]))
            seen_targets.add(target)
            self.ctc_feasible_count += int(t["ctc_feasible_40hz"])

        excluded = [row["target_key"] for row in _read_jsonl(excluded_path)]
        if (
            len(self.indices) != summary["accepted"]
            or not self.indices
            or len(excluded) != summary["excluded"]
            or len(seen_targets) + len(excluded) != len(records)
            or seen_targets.union(excluded) != by_key.keys()
        ):
            raise ValueError("Accepted and excluded targets must partition the entire training manifest exactly once")
        self.ctc_infeasible_count = len(self.indices) - self.ctc_feasible_count
        # Binding this into the run contract prevents resuming with a different map.
        self.pairing_contract = {
            "summary_sha256": sha256_file(summary_path),
            **{
                field: summary[field]
                for field in (
                    "train_manifest_sha256",
                    "pairs_sha256",
                    "exclusions_sha256",
                    "accepted",
                    "excluded",
                    "settings",
                )
            },
        }

    def __len__(self):
        return len(self.indices)

    def get_frame_len(self, index):
        t, r = self.indices[index]
        return self.base_dataset.get_frame_len(t) + self.base_dataset.get_frame_len(r)

    def __getitem__(self, index):
        base = self.base_dataset
        t, r = self.indices[index]
        target, reference = base.records[t], base.records[r]
        item = base._target_item(target)
        prompt = base._normalized_latent(reference)
        p = prompt.size(1)
        item.update(
            mel_spec=torch.cat((prompt, item["mel_spec"]), dim=1),
            video=F.pad(item["video"], (0, 0, p, 0)),
            speaker_embedding=torch.from_numpy(base._load_speaker_embedding_array(reference)),
            reference_text=reference["text"],
            reference_utterance_key=reference["utterance_key"],
            prompt_len=p,
        )
        if base.repa_feature_cache_dir is not None:
            item["repa_feature"] = torch.from_numpy(base._load_repa_feature_array(target))
        return item

    @staticmethod
    def collate_fn(batch):
        result = SemanticVaeCelebVDubDataset.collate_fn(batch)
        result.update(
            reference_text=[item["reference_text"] for item in batch],
            reference_utterance_keys=[item["reference_utterance_key"] for item in batch],
            prompt_lens=torch.tensor([item["prompt_len"] for item in batch], dtype=torch.long),
        )
        return result
