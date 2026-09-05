"""D1 trainer with strict loading of the compatible Semantic-VAE S2c EMA."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch

from aligndit.model.semantic_vae_direct_migration import (
    load_s2c_ema_state,
    migrate_s2c_ema_into_model,
    validate_parent_artifacts,
)
from aligndit.model.trainer_vt import Trainer_VT


class SemanticVaeDirectD1Trainer(Trainer_VT):
    """Preserve D1 optimization while enforcing the Semantic-VAE parent migration."""

    def __init__(
        self,
        *args,
        parent_contract_path: str,
        expected_parent_sha256: str,
        expected_parent_size: int,
        expected_parent_contract_sha256: str,
        expected_parent_update: int = 70_000,
        **kwargs,
    ):
        self.parent_contract_path = parent_contract_path
        self.expected_parent_sha256 = expected_parent_sha256
        self.expected_parent_size = expected_parent_size
        self.expected_parent_contract_sha256 = expected_parent_contract_sha256
        self.expected_parent_update = expected_parent_update
        super().__init__(*args, **kwargs)

    def load_pretrained(self, pretrained_path):
        """Load exactly the compatible S2c EMA tensors into online and EMA models."""

        self.accelerator.wait_for_everyone()
        if self.is_main:
            validate_parent_artifacts(
                pretrained_path,
                self.parent_contract_path,
                expected_checkpoint_sha256=self.expected_parent_sha256,
                expected_checkpoint_size=self.expected_parent_size,
                expected_contract_sha256=self.expected_parent_contract_sha256,
            )
        self.accelerator.wait_for_everyone()

        source_state, parent_ema_step = load_s2c_ema_state(
            pretrained_path,
            expected_parent_contract_sha256=self.expected_parent_contract_sha256,
            expected_parent_update=self.expected_parent_update,
        )
        online_model = self.accelerator.unwrap_model(self.model)
        online_report = migrate_s2c_ema_into_model(
            online_model,
            source_state,
            parent_path=pretrained_path,
            parent_sha256=self.expected_parent_sha256,
            parent_size=self.expected_parent_size,
            parent_contract_sha256=self.expected_parent_contract_sha256,
            parent_ema_step=parent_ema_step,
        )

        if self.is_main:
            ema_report = migrate_s2c_ema_into_model(
                self.ema_model.ema_model,
                source_state,
                parent_path=pretrained_path,
                parent_sha256=self.expected_parent_sha256,
                parent_size=self.expected_parent_size,
                parent_contract_sha256=self.expected_parent_contract_sha256,
                parent_ema_step=parent_ema_step,
            )
            if online_report != ema_report:
                raise RuntimeError("Online and EMA S2c migration reports differ")
            self.ema_model.initted.fill_(True)
            self.ema_model.step.fill_(parent_ema_step)

            os.makedirs(self.checkpoint_path, exist_ok=True)
            report_path = Path(self.checkpoint_path) / "parent_migration.json"
            temporary_path = report_path.with_suffix(".json.tmp")
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(online_report.to_dict(), file, ensure_ascii=False, indent=2, sort_keys=True)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, report_path)
            print(
                "Strict S2c EMA migration passed: "
                f"source={online_report.source_key_count}, target={online_report.target_key_count}, "
                f"loaded={online_report.loaded_key_count}, ignored={len(online_report.ignored_source_keys)}, "
                f"new={len(online_report.new_target_keys)}, EMA step={parent_ema_step}"
            )

        del source_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
