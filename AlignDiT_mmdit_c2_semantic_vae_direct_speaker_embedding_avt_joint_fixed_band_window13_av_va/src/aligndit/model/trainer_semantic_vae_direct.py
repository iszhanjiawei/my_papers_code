"""C2 training with audited S2c or LibriTTS EMA initialization."""

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


class SemanticVaeDirectC2Trainer(Trainer_VT):
    """Preserve C2 optimization while validating the selected parent migration."""

    def __init__(
        self,
        *args,
        parent_contract_path: str,
        expected_parent_sha256: str,
        expected_parent_size: int,
        expected_parent_contract_sha256: str,
        expected_parent_update: int = 70_000,
        initialization_mode: str = "s2c",
        text_audio_contract: dict | None = None,
        allow_layer_repartition: bool = False,
        **kwargs,
    ):
        if initialization_mode not in {"s2c", "libritts_text_audio"}:
            raise ValueError(f"Unknown initialization mode: {initialization_mode}")
        self.initialization_mode = initialization_mode
        self.text_audio_contract = text_audio_contract
        self.allow_layer_repartition = allow_layer_repartition
        self.parent_contract_path = parent_contract_path
        self.expected_parent_sha256 = expected_parent_sha256
        self.expected_parent_size = expected_parent_size
        self.expected_parent_contract_sha256 = expected_parent_contract_sha256
        self.expected_parent_update = expected_parent_update
        super().__init__(*args, **kwargs)

    def load_pretrained(self, pretrained_path):
        """Initialize online and EMA models using the selected explicit migration."""

        if self.initialization_mode == "libritts_text_audio":
            return self._load_text_audio_pretrained(pretrained_path)

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

    def _load_text_audio_pretrained(self, pretrained_path):
        from aligndit.model.text_audio_checkpoint import load_text_audio_pretrained

        if self.text_audio_contract is None:
            raise ValueError("LibriTTS initialization requires the downstream representation contract")

        def collective_error(error):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                errors = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(errors, error)
            else:
                errors = [error]
            if any(errors):
                raise RuntimeError("LibriTTS initialization failed: " + "; ".join(item for item in errors if item))

        online = self.accelerator.unwrap_model(self.model)
        error, report = None, None
        try:
            report = load_text_audio_pretrained(
                online, pretrained_path, self.text_audio_contract,
                allow_layer_repartition=self.allow_layer_repartition,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        collective_error(error)
        error = None
        if self.is_main:
            try:
                self.ema_model.ema_model.load_state_dict(online.state_dict(), strict=True)
                self.ema_model.initted.fill_(True)
                self.ema_model.step.zero_()
                destination = Path(self.checkpoint_path)
                destination.mkdir(parents=True, exist_ok=True)
                temporary = destination / "parent_migration.json.tmp"
                temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                os.replace(temporary, destination / "parent_migration.json")
                print(
                    "LibriTTS EMA transfer passed: "
                    f"source_update={report['source_update']}, "
                    f"joint_layers={report['source_joint_layers']}->{report['target_joint_layers']}, "
                    f"loaded={report['loaded_tensors']}, "
                    f"discarded_stream_tensors={len(report['discarded_stream_tensors'])}; "
                    f"full report={destination / 'parent_migration.json'}",
                    flush=True,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        collective_error(error)
        self.accelerator.wait_for_everyone()
