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
        # Concurrency guard: refuse if a training job is already running
        with self._lock:
            if any(j.status == "running" for j in self._jobs.values()):
                logger.info("[lora] training already in progress — skipping")
                return None
        # RAM gate: require ≥ 4 GB free to load a quantised model
        try:
            import psutil as _ps
            if _ps.virtual_memory().available < 4 * 1024 ** 3:
                logger.info("[lora] <4 GB RAM free — skipping training")
                return None
        except Exception:
            pass
        # Phase gate: defer if system is under thermal/RAM pressure
        try:
            import prism_phase as _pp
            _engine = _pp.get_engine()
            if _engine.history and _engine.current_phase.value in ("VISCOUS", "LIQUID"):
                logger.info("[lora] phase=%s — skipping training", _engine.current_phase.value)
                return None
        except Exception:
            pass
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
        """Aggregate DPO preference pairs from two sources:

        1. ``OutcomeRecord.correction`` — chains where the user supplied
           a corrected answer. ``rejected`` = the model's original output.
        2. ``PrismInstructions`` denial-derived rules (M12c) — both
           task-scoped one-shot denials and broad standing rules become
           preference pairs so the user's "no" or "never" actively trains
           the personalised LoRA, not just guards the runtime gate.

        Each pair: ``{"prompt": str, "chosen": str, "rejected": str}``.
        Both sources are best-effort; missing DBs are silent.
        """
        pairs: list[dict] = []

        # 1. Outcome corrections
        try:
            from prism_outcome_tracker import OutcomeTracker
            tracker_path = Path("~/.prism/outcomes.db").expanduser()
            if tracker_path.exists():
                tracker = OutcomeTracker(db_path=str(tracker_path))
                for r in tracker.recent(n=500):
                    if getattr(r, "correction", None) and getattr(r, "final_answer", None):
                        pairs.append({
                            "prompt":   getattr(r, "goal", ""),
                            "chosen":   r.correction,
                            "rejected": r.final_answer,
                        })
        except Exception as exc:
            logger.debug("[lora] Could not load outcome records: %s", exc)

        # 2. Denial-derived standing / task-scoped rules
        try:
            from prism_instructions import PrismInstructions
            instr_path = Path("~/.prism/instructions.db").expanduser()
            if instr_path.exists():
                instr = PrismInstructions(db_path=str(instr_path))
                pairs.extend(self._pairs_from_instructions(instr))
        except Exception as exc:
            logger.debug("[lora] Could not load denial instructions: %s", exc)

        return pairs

    def _pairs_from_instructions(self, instr) -> list[dict]:
        """Translate every active stored rule into a DPO preference pair."""
        out: list[dict] = []
        try:
            rules = instr.all_active()
        except Exception:
            return out
        for rule in rules:
            pair = self._denial_to_dpo_pair(rule)
            if pair is not None:
                out.append(pair)
        return out

    @staticmethod
    def _denial_to_dpo_pair(rule) -> Optional[dict]:
        """Translate one stored ``Instruction`` into a DPO pair.

        Three flavours, distinguished by ``rule.trigger``:

        * Standing rule on a TRIGGER_MAP category (email / calendar / …)
          → prompt asks for the user's rule for that category.
        * Task-slug trigger → prompt phrases the original request and asks
          whether to proceed, so the rule trains "decline this kind."
        * ``"always"`` trigger → prompt asks for the universal rule.

        Returns ``None`` for empty / whitespace-only text. ``rejected`` is
        a generic permissive completion the LoRA must learn to dispreference
        in favour of the user's explicit rule.
        """
        from prism_instructions import PrismInstructions
        text = (getattr(rule, "text", "") or "").strip()
        if not text:
            return None
        trig = getattr(rule, "trigger", "") or ""
        if trig == "always":
            prompt   = "What rule should I always follow when assisting this user?"
            rejected = "No specific universal rule applies — I'll act on the request as stated."
        elif trig in PrismInstructions.TRIGGER_MAP:
            prompt   = f"What is the user's standing rule for {trig} requests?"
            rejected = f"I don't have a specific rule for {trig} — I'll act on the request as stated."
        else:
            # Task-slug trigger: a one-shot denial guard.
            human = trig.replace("_", " ").replace("-", " ").strip() or "this task"
            prompt   = f"The user asks me to {human}. Should I proceed?"
            rejected = "Yes, proceeding with the task."
        return {"prompt": prompt, "chosen": text, "rejected": rejected}

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
