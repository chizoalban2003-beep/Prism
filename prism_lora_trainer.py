"""
prism_lora_trainer.py — LoRA QLoRA training pipeline for PRISM.
OutcomeRecord corrections → DPO pairs → Unsloth QLoRA → GGUF → Ollama.
Stubs gracefully when unsloth/trl are absent (CI safe).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    job_id: str
    base_model: str           # e.g. "llama3.2:3b"
    status: str               # "pending" | "running" | "done" | "failed"
    started_at: float
    finished_at: Optional[float] = None
    error: Optional[str] = None
    gguf_path: Optional[str] = None
    ollama_model: Optional[str] = None
    pairs_used: int = 0


class PrismLoraTrainer:
    """
    Orchestrates: OutcomeRecord corrections → DPO pairs → Unsloth QLoRA →
    GGUF export → Ollama registration.

    Runs training in a background thread so the daemon stays responsive.
    """

    def __init__(self, work_dir: str = "~/.prism/lora") -> None:
        self._work_dir = Path(work_dir).expanduser()
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def start_training(self, base_model: str = "llama3.2:3b", min_pairs: int = 10) -> Optional[str]:
        """Collect DPO pairs, launch background training. Returns job_id or None."""
        pairs = self._collect_dpo_pairs()
        if len(pairs) < min_pairs:
            logger.info("[lora] Only %d pairs (need %d), skipping", len(pairs), min_pairs)
            return None
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(
            job_id=job_id, base_model=base_model, status="pending",
            started_at=time.time(), pairs_used=len(pairs),
        )
        with self._lock:
            self._jobs[job_id] = job
        threading.Thread(
            target=self._run_training, args=(job, pairs),
            daemon=True, name=f"lora-{job_id}",
        ).start()
        return job_id

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[TrainingJob]:
        with self._lock:
            return list(self._jobs.values())

    # ------------------------------------------------------------------

    def _collect_dpo_pairs(self) -> list[dict]:
        """Pull DPO pairs from OutcomeRecord.correction field.
        Each pair: {"prompt": str, "chosen": str, "rejected": str}
        """
        pairs: list[dict] = []
        try:
            from prism_outcome_tracker import OutcomeTracker
            tracker_path = Path("~/.prism/outcomes.db").expanduser()
            if not tracker_path.exists():
                return []
            tracker = OutcomeTracker(db_path=str(tracker_path))
            for r in tracker.recent(limit=500):
                if getattr(r, "correction", None) and getattr(r, "final_answer", None):
                    pairs.append({
                        "prompt":   getattr(r, "goal", ""),
                        "chosen":   r.correction,
                        "rejected": r.final_answer,
                    })
        except Exception as exc:
            logger.debug("[lora] Could not load outcome records: %s", exc)
        return pairs

    def _run_training(self, job: TrainingJob, pairs: list[dict]) -> None:
        """Background thread: write pairs → train → Ollama register."""
        job.status = "running"
        try:
            pairs_path = self._work_dir / f"{job.job_id}_pairs.jsonl"
            with pairs_path.open("w") as fh:
                for p in pairs:
                    fh.write(json.dumps(p) + "\n")

            gguf_path = self._train_unsloth(job, pairs_path)
            if gguf_path is None:
                raise RuntimeError("Unsloth training returned no output")

            ollama_name = self._register_ollama(job.job_id, gguf_path, job.base_model)

            try:
                from prism_lora_registry import LoRARegistry
                LoRARegistry().register(job.job_id, str(gguf_path), ollama_name)
            except Exception as exc:
                logger.warning("[lora] Registry update failed: %s", exc)

            job.gguf_path = str(gguf_path)
            job.ollama_model = ollama_name
            job.status = "done"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            logger.warning("[lora] Training job %s failed: %s", job.job_id, exc)
        finally:
            job.finished_at = time.time()

    def _train_unsloth(self, job: TrainingJob, pairs_path: Path) -> Optional[Path]:
        """Unsloth QLoRA → GGUF. Stubs with a placeholder when unsloth absent."""
        output_dir = self._work_dir / f"{job.job_id}_output"
        output_dir.mkdir(exist_ok=True)
        try:
            import torch
            from trl import DPOConfig, DPOTrainer
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=job.base_model, max_seq_length=2048, load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model, r=16, target_modules=["q_proj", "v_proj"],
                lora_alpha=16, lora_dropout=0, bias="none",
            )
            from datasets import Dataset
            data = [json.loads(ln) for ln in pairs_path.read_text().splitlines() if ln.strip()]
            trainer = DPOTrainer(
                model=model,
                args=DPOConfig(
                    output_dir=str(output_dir), num_train_epochs=1,
                    per_device_train_batch_size=2,
                    bf16=torch.cuda.is_available(), logging_steps=10, report_to="none",
                ),
                train_dataset=Dataset.from_list(data),
                tokenizer=tokenizer,
            )
            trainer.train()
            model.save_pretrained_gguf(str(output_dir), tokenizer, quantization_method="q4_k_m")
            candidates = list(output_dir.glob("*.gguf"))
            return candidates[0] if candidates else None
        except ImportError:
            logger.info("[lora] unsloth/trl not installed — writing placeholder GGUF")
            stub = output_dir / "model.gguf"
            stub.write_bytes(b"GGUF_STUB")
            return stub

    def _register_ollama(self, job_id: str, gguf_path: Path, base_model: str) -> str:
        """Write Ollama Modelfile and register via REST API."""
        import urllib.request
        model_name = f"prism-lora-{job_id}"
        modelfile = (
            f"FROM {gguf_path}\n"
            "SYSTEM You are PRISM, a local-first AI assistant personalised to this user.\n"
        )
        (gguf_path.parent / "Modelfile").write_text(modelfile)
        payload = json.dumps({"name": model_name, "modelfile": modelfile}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/create",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                logger.info("[lora] Ollama registered %s: %s", model_name, resp.status)
        except Exception as exc:
            logger.warning("[lora] Ollama registration failed (continuing): %s", exc)
        return model_name
