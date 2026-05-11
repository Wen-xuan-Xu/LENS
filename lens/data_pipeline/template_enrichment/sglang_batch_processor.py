"""High-concurrency LLM enrichment client (SGLang / OpenAI-compatible HTTP).

Sends the rule-based EMA templates to a locally served instruction-tuned LLM
(Qwen2.5-14B in the paper, exposed via SGLang's OpenAI-compatible API) and
collects the rewritten narratives.  Two modes are supported:

  * ``process_narratives_concurrently`` -- rewrite ``template_narrative`` for
    each record of the templates JSON file; one output row per record.
  * ``process_sampled_concurrently`` -- rewrite each sampled item response in a
    record's ``question_responses`` dict; one output row per (record, item).

The ``requests`` package is imported lazily so this module can be imported (and
the dataclasses re-used) even when the optional ``pipeline`` extra is not
installed.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional

from .prompt import create_enrichment_prompt

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for the concurrent enrichment client."""

    sglang_url: str = "http://localhost:30000"
    model_name: str = "qwen2.5-14b"
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    max_workers: int = 16


def _safe_get_qa_text(item: Dict, qkey: str) -> str:
    """Return the rule-based answer text for a question key (e.g. ``ema_Q12``)."""
    qresp = item.get("question_responses") or {}
    return (qresp.get(qkey) or "").strip()


class SGLangConcurrentProcessor:
    """Concurrent request processor against an OpenAI-compatible LLM server."""

    def __init__(self, config: BatchConfig):
        self.config = config

    # ------------------------------------------------------------------ #
    def check_server(self) -> bool:
        """Return True if the server's ``/v1/models`` endpoint responds 200."""
        import requests  # lazy: optional dependency

        try:
            resp = requests.get(f"{self.config.sglang_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.error("Server check failed: %s", exc)
            return False

    def _execute_single_request(self, payload: Dict, timeout: int = 30) -> Optional[Dict]:
        import requests  # lazy: optional dependency

        try:
            url = f"{self.config.sglang_url}/v1/chat/completions"
            req = {
                "model": payload["model"],
                "messages": payload["messages"],
                "temperature": payload["temperature"],
                "max_tokens": payload["max_tokens"],
                "top_p": self.config.top_p,
                "stop": payload.get("stop", []),
                "chat_template_kwargs": {"enable_thinking": False},
            }
            resp = requests.post(url, json=req, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            enhanced_text = ""
            if result.get("choices"):
                msg = result["choices"][0].get("message", {})
                enhanced_text = (msg.get("content") or "").strip()
            return {
                "group_id": payload["group_id"],
                "task_type": payload["task_type"],
                "question_id": payload.get("question_id"),
                "enhanced_text": enhanced_text,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Request failed for %s: %s", payload.get("custom_id"), exc)
            return None

    # ------------------------------------------------------------------ #
    def process_narratives_concurrently(self, input_json_file: str, output_jsonl: str,
                                        record_offset: int = 0,
                                        record_limit: Optional[int] = None,
                                        timeout: int = 30) -> None:
        """Rewrite the ``template_narrative`` field of every record."""
        with open(input_json_file, "r", encoding="utf-8") as f:
            all_records = json.load(f)
        records = all_records[record_offset:]
        if record_limit:
            records = records[:record_limit]
        if not records:
            logger.info("No records to process.")
            return

        payloads = []
        for i, item in enumerate(records):
            payloads.append({
                "custom_id": f"narr_{record_offset + i}",
                "group_id": record_offset + i,
                "task_type": "narrative",
                "model": self.config.model_name,
                "messages": create_enrichment_prompt(item.get("template_narrative", "")),
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stop": ["**", "Input:", "Output:", "Example"],
                "uid": item.get("uid"),
                "ema_timestamp": item.get("ema_timestamp"),
                "rule_based_narrative": item.get("template_narrative", ""),
            })

        with open(output_jsonl, "w"):
            pass
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
            futures = {ex.submit(self._execute_single_request, p, timeout): p for p in payloads}
            for fut in as_completed(futures):
                res = fut.result()
                payload = futures[fut]
                if not res:
                    continue
                out = {
                    "uid": payload.get("uid"),
                    "ema_timestamp": payload.get("ema_timestamp"),
                    "rule_based_narrative": payload.get("rule_based_narrative", ""),
                    "enhanced_narrative": res.get("enhanced_text", ""),
                    "generation_params": {
                        "model": self.config.model_name,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    },
                }
                with open(output_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

    def process_sampled_concurrently(self, input_json_file: str, output_jsonl: str,
                                     record_offset: int = 0,
                                     record_limit: Optional[int] = None,
                                     timeout: int = 30) -> None:
        """Rewrite each item-level answer in every record's ``question_responses``."""
        with open(input_json_file, "r", encoding="utf-8") as f:
            all_records = json.load(f)
        records = all_records[record_offset:]
        if record_limit:
            records = records[:record_limit]
        if not records:
            logger.info("No records to process.")
            return

        payloads = []
        task_id = 0
        for item in records:
            uid, ts = item.get("uid"), item.get("ema_timestamp")
            for qkey, qtext in (item.get("question_responses") or {}).items():
                payloads.append({
                    "custom_id": f"sampled_{task_id}",
                    "group_id": task_id,
                    "task_type": "qa",
                    "question_id": qkey,
                    "model": self.config.model_name,
                    "messages": create_enrichment_prompt(qtext),
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "stop": ["**", "Input:", "Output:", "Example"],
                    "uid": uid,
                    "ema_timestamp": ts,
                    "rule_based_answer": qtext,
                    "question_key": qkey,
                })
                task_id += 1

        with open(output_jsonl, "w"):
            pass
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
            futures = {ex.submit(self._execute_single_request, p, timeout): p for p in payloads}
            for fut in as_completed(futures):
                res = fut.result()
                payload = futures[fut]
                if not res:
                    continue
                out = {
                    "uid": payload.get("uid"),
                    "ema_timestamp": payload.get("ema_timestamp"),
                    "question_key": payload.get("question_key"),
                    "rule_based_answer": payload.get("rule_based_answer", ""),
                    "enhanced_answer": res.get("enhanced_text", ""),
                    "generation_params": {
                        "model": self.config.model_name,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    },
                }
                with open(output_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
