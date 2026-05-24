import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

from src.latent_pipeline.common import log


@dataclass
class LLMConfig:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    max_input_tokens: int = 3072


class LLMBackend:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.utils import logging as hf_logging
        except Exception as e:
            raise RuntimeError(f"LLM backend requires transformers+torch: {e}")

        self.torch = torch
        # Silence repetitive transformers warnings in slurm logs.
        hf_logging.set_verbosity_error()
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=hf_token)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model_kwargs = {
            "token": hf_token,
            "torch_dtype": (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                # Spread large checkpoints across visible GPUs and avoid concentrating layers on cuda:0.
                model_kwargs["device_map"] = "balanced_low_0"
                max_memory = {}
                for i in range(n_gpu):
                    tot_gib = int(torch.cuda.get_device_properties(i).total_memory // (1024**3))
                    # Keep a small headroom per device for runtime allocations.
                    max_memory[i] = f"{max(8, tot_gib - 6)}GiB"
                model_kwargs["max_memory"] = max_memory
            else:
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"
        # Some large checkpoints require explicit disk offload folder when auto/balanced mapping spills.
        offload_root = os.environ.get("HF_OFFLOAD_DIR") or os.path.join(
            os.environ.get("TMPDIR", "/tmp"),
            "trace_hf_offload",
        )
        offload_dir = Path(offload_root) / re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(cfg.model_id))
        offload_dir.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(offload_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            **model_kwargs,
        )
        if self.model.generation_config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.generation_config.eos_token_id is None and self.tokenizer.eos_token_id is not None:
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self._calls = 0
        self._failures = 0
        self._raw_logged = 0
        self._raw_log_limit = int(os.environ.get("LLM_DEBUG_RAW_N", "0") or 0)
        log(f"LLM backend initialized: model={cfg.model_id} device={self.model.device}")

    def _build_prompt_text(self, system_prompt: str, user_prompt: str) -> str:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Preferred path for instruct/chat models.
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # Fallback for base/non-chat tokenizers without chat_template.
        return (
            "System:\n"
            f"{system_prompt}\n\n"
            "User:\n"
            f"{user_prompt}\n\n"
            "Assistant:\n"
        )

    def _extract_json(self, txt: str) -> Dict[str, Any]:
        txt = txt.strip()
        # 1) Direct JSON object extraction.
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            js = m.group(0)
            # best-effort cleanup for common LLM JSON issues
            js2 = re.sub(r",\s*([}\]])", r"\1", js)  # trailing comma
            js2 = js2.replace("'", "\"")
            for cand in (js, js2):
                try:
                    return json.loads(cand)
                except Exception:
                    pass

        # 2) Key-value fallback extraction for probability blobs.
        labels = ["vasopressor_signal", "resp_support_signal", "renal_support_signal", "any_deterioration"]
        out: Dict[str, Any] = {}
        for k in labels:
            m2 = re.search(
                rf"[\"']?{re.escape(k)}[\"']?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                txt,
                flags=re.IGNORECASE,
            )
            if m2:
                try:
                    v = float(m2.group(1))
                    out[k] = max(0.0, min(1.0, v))
                except Exception:
                    continue
        if out:
            return out
        return {}

    def _extract_harmony_final(self, txt: str) -> str:
        """Extract GPT-OSS Harmony final channel payload when present."""
        if not txt:
            return ""
        patterns = [
            r"<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\|end\|>|$)",
            r"<\|start\|>assistant<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\|end\|>|$)",
        ]
        for pat in patterns:
            m = re.search(pat, txt, flags=re.IGNORECASE)
            if m:
                return (m.group(1) or "").strip()
        return ""

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        text = self._build_prompt_text(system_prompt, user_prompt)
        self._calls += 1
        try:
            model_id_l = str(self.cfg.model_id).lower()
            is_gpt_oss = "gpt-oss" in model_id_l
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.max_input_tokens,
            ).to(self.model.device)
            max_new_tokens = int(self.cfg.max_new_tokens)
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    do_sample=(self.cfg.temperature > 0),
                    use_cache=(False if is_gpt_oss else True),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            out_tokens = out[0][inputs["input_ids"].shape[1]:]
            gen = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
            gen_raw = self.tokenizer.decode(out_tokens, skip_special_tokens=False)
            # GPT-OSS can emit analysis/final channels; prefer final payload if available.
            preferred = self._extract_harmony_final(gen_raw) if is_gpt_oss else ""
            parsed = self._extract_json(preferred or gen)
            if self._raw_logged < self._raw_log_limit:
                self._raw_logged += 1
                log(
                    f"LLM raw[{self._raw_logged}/{self._raw_log_limit}] "
                    f"model={self.cfg.model_id} text={repr((preferred or gen)[:1000])} parsed={parsed}"
                )
            return parsed
        except Exception as e:
            self._failures += 1
            if self._failures <= 5:
                log(f"LLM generate_json failure[{self._failures}] model={self.cfg.model_id}: {type(e).__name__}: {e}")
            # Return empty JSON so caller can safely fallback to deterministic behavior.
            return {}
