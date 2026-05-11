"""Parallel multi-agent LLM-as-a-judge.

Deploys all judge models simultaneously (one per GPU) and evaluates every
narrative with each model concurrently, then combines the per-model scores with
confidence-weighted averaging into a single PASS/FAIL verdict.  Use
:mod:`sequential_judge` instead when GPU memory is tight.

Input JSONL records must contain ``rule_based_narrative`` and
``llm_enriched_narrative``.  Heavy / optional dependencies (``openai``,
``aiohttp``) are imported lazily so this module is importable without the
``pipeline`` extra.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .judge_config import (JUDGE_MODELS_AVAILABLE, JUDGE_SYSTEM_PROMPT,
                           JUDGE_USER_PROMPT_TEMPLATE, ModelConfig, VotingConfig)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PASS_THRESHOLD = VotingConfig().pass_threshold


@dataclass
class ParallelJudgeConfig:
    """Configuration for the parallel judge run."""

    input_file: str
    output_dir: str = "./parallel_judge_results"
    batch_size: int = 16
    max_records: int = 0  # 0 = process all
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 180
    models_to_run: Optional[List[str]] = None  # None = all available
    deploy: bool = True  # If False, assume models are already serving.


class ParallelJudgeSystem:
    """Parallel LLM judge system -- deploy all models simultaneously."""

    def __init__(self, config: ParallelJudgeConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if config.models_to_run:
            self.models = [m for m in JUDGE_MODELS_AVAILABLE if m.model_name in config.models_to_run]
        else:
            self.models = JUDGE_MODELS_AVAILABLE.copy()
        logger.info("Models to run in parallel: %s", [m.model_name for m in self.models])

    # ------------------------------------------------------------------ #
    async def run_parallel_evaluation(self):
        input_data = self._load_input_data()
        if not input_data:
            logger.error("No input data loaded")
            return
        logger.info("Loaded %d records for evaluation", len(input_data))

        if self.config.deploy:
            logger.info("Deploying %d models in parallel...", len(self.models))
            results = await asyncio.gather(
                *[self._deploy_single_model(m, i) for i, m in enumerate(self.models)],
                return_exceptions=True)
            active_models = [m for m, r in zip(self.models, results)
                             if not isinstance(r, BaseException) and r]
        else:
            active_models = self.models

        if len(active_models) < 2:
            logger.error("Only %d models available, need at least 2", len(active_models))
            return
        logger.info("Starting parallel evaluation with %d models", len(active_models))

        try:
            model_results = await asyncio.gather(
                *[self._evaluate_with_single_model(m, input_data) for m in active_models],
                return_exceptions=True)
            successful: Dict[str, List[Dict]] = {}
            for model, res in zip(active_models, model_results):
                if isinstance(res, BaseException):
                    logger.error("Evaluation failed for %s: %s", model.model_name, res)
                    continue
                successful[model.model_name] = res
                out_file = self.output_dir / f"{model.model_name}_evaluations.jsonl"
                self._save_jsonl(res, out_file)
                logger.info("Saved %s results to %s", model.model_name, out_file)

            if len(successful) >= 2:
                combined = self._combine_model_results(input_data, successful)
                combined_file = self.output_dir / "combined_judge_results.jsonl"
                self._save_jsonl(combined, combined_file)
                logger.info("Combined results saved to %s", combined_file)
                self._generate_summary_report(successful, combined)
            else:
                logger.warning("Fewer than 2 models completed; no combined results")
        finally:
            if self.config.deploy:
                await asyncio.gather(*[self._cleanup_model(m) for m in active_models],
                                     return_exceptions=True)

    # ------------------------------------------------------------------ #
    def _load_input_data(self) -> List[Dict]:
        data: List[Dict] = []
        try:
            with open(self.config.input_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("Line %d: JSON decode error: %s", line_num + 1, exc)
                        continue
                    if "rule_based_narrative" in obj and "llm_enriched_narrative" in obj:
                        data.append(obj)
                    else:
                        logger.warning("Line %d: missing required narrative fields", line_num + 1)
                    if self.config.max_records > 0 and len(data) >= self.config.max_records:
                        break
        except FileNotFoundError:
            logger.error("Input file not found: %s", self.config.input_file)
        return data

    async def _deploy_single_model(self, model_config: ModelConfig, gpu_id: int) -> bool:
        logger.info("Deploying %s at %s on GPU %d", model_config.model_name,
                    model_config.service_url, gpu_id)
        cmd = ["python3", "-m", "sglang.launch_server",
               "--model-path", model_config.model_path,
               "--host", "0.0.0.0", "--port", str(model_config.port),
               "--mem-fraction-static", "0.6"]
        if model_config.extra_args:
            cmd.extend(model_config.extra_args.split())
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            subprocess.Popen(cmd, env=env)  # noqa: S603
            start = time.time()
            while time.time() - start < 300:
                if await self._check_server_ready(model_config.service_url):
                    logger.info("%s is ready", model_config.model_name)
                    return True
                await asyncio.sleep(10)
            logger.error("%s failed to start within 300s", model_config.model_name)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to start %s: %s", model_config.model_name, exc)
            return False

    async def _check_server_ready(self, service_url: str) -> bool:
        import aiohttp  # lazy: optional dependency

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{service_url}/v1/models",
                                       timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except Exception:  # noqa: BLE001
            return False

    async def _evaluate_with_single_model(self, model_config: ModelConfig,
                                          input_data: List[Dict]) -> List[Dict]:
        from openai import AsyncOpenAI  # lazy: optional dependency

        logger.info("Evaluating %d records with %s", len(input_data), model_config.model_name)
        client = AsyncOpenAI(base_url=f"{model_config.service_url}/v1", api_key="dummy")
        results: List[Dict] = []
        for i in range(0, len(input_data), self.config.batch_size):
            batch = input_data[i:i + self.config.batch_size]
            batch_results = await asyncio.gather(
                *[self._evaluate_single_item(client, item, model_config.model_name) for item in batch],
                return_exceptions=True)
            for orig, res in zip(batch, batch_results):
                if isinstance(res, BaseException):
                    results.append({**orig,
                                    f"{model_config.model_name}_evaluation":
                                        {"error": str(res), "total_score": 5,
                                         "avg_confidence": 0.0, "valid": False},
                                    "evaluation_timestamp": time.time()})
                else:
                    results.append({**orig, f"{model_config.model_name}_evaluation": res,
                                    "evaluation_timestamp": time.time()})
        return results

    async def _evaluate_single_item(self, client, item: Dict, model_name: str) -> Dict:
        try:
            prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
                original_template=item["rule_based_narrative"],
                enriched_narrative=item["llm_enriched_narrative"])
            response = await client.chat.completions.create(
                model="default",
                messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=self.config.temperature, max_tokens=self.config.max_tokens,
                timeout=self.config.timeout)
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("Empty response content")
            return self._parse_evaluation(content, model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("API request failed for %s: %s", model_name, exc)
            return {"error": f"API error: {exc}", "total_score": 5,
                    "avg_confidence": 0.0, "model_name": model_name, "valid": False}

    @staticmethod
    def _parse_evaluation(content: str, model_name: str) -> Dict:
        try:
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.strip("`\n ")
            if not (content.startswith("{") and content.endswith("}")):
                start, end = content.find("{"), content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    content = content[start:end + 1]
            ev = json.loads(content)
            if (isinstance(ev.get("scores"), list) and len(ev["scores"]) == 5
                    and isinstance(ev.get("confidence"), list) and len(ev["confidence"]) == 5):
                return {"scores": ev["scores"], "confidence": ev["confidence"],
                        "critique": ev.get("critique", {}),
                        "total_score": sum(ev["scores"]),
                        "avg_confidence": round(statistics.mean(ev["confidence"]), 3),
                        "model_name": model_name, "valid": True}
            raise ValueError("Invalid evaluation format")
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse evaluation from %s: %s", model_name, exc)
            return {"error": f"JSON parse error: {exc}", "raw_content": content[:500],
                    "total_score": 5, "avg_confidence": 0.0,
                    "model_name": model_name, "valid": False}

    async def _cleanup_model(self, model_config: ModelConfig):
        try:
            result = subprocess.run(["pgrep", "-f", f"port {model_config.port}"],  # noqa: S603,S607
                                    capture_output=True, text=True)
            for pid in result.stdout.strip().split("\n"):
                if pid:
                    subprocess.run(["kill", "-9", pid], check=False)  # noqa: S603,S607
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cleanup warning for %s: %s", model_config.model_name, exc)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _save_jsonl(records: List[Dict], output_file: Path):
        with open(output_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _combine_model_results(self, input_data: List[Dict],
                               model_results: Dict[str, List[Dict]]) -> List[Dict]:
        combined: List[Dict] = []
        for i, orig in enumerate(input_data):
            evals = []
            for model_name, result_list in model_results.items():
                if i < len(result_list):
                    ev = result_list[i].get(f"{model_name}_evaluation", {})
                    if ev.get("valid"):
                        evals.append(ev)
            if len(evals) < 2:
                logger.warning("Item %d: only %d valid evaluations, skipping", i, len(evals))
                continue
            ce = self._average_evaluations(evals)
            agreement = self._calculate_agreement(evals)
            ce["consensus_quality"] = self._classify_consensus_quality(ce["overall_confidence"], agreement)
            ce["quality_summary"] = (f"Combined score: {ce['total_score']}/25. "
                                     f"Quality: {ce['overall_quality']}. "
                                     f"Avg Confidence: {ce['overall_confidence']}. "
                                     f"Consensus: {ce['consensus_quality']}.")
            combined.append({**orig, "combined_evaluation": ce,
                             "individual_evaluations": {e["model_name"]: e for e in evals},
                             "agreement_level": agreement, "num_models": len(evals),
                             "evaluation_timestamp": time.time()})
        return combined

    @staticmethod
    def _average_evaluations(evaluations: List[Dict]) -> Dict:
        all_scores = [e["scores"] for e in evaluations if "scores" in e]
        all_conf = [e["confidence"] for e in evaluations if "confidence" in e]
        if all_scores and all_conf and len(all_scores) == len(all_conf):
            avg_scores = []
            for k in range(5):
                scores = [s[k] for s in all_scores]
                confs = [c[k] for c in all_conf]
                if sum(confs) > 0:
                    avg_scores.append(round(sum(s * c for s, c in zip(scores, confs)) / sum(confs)))
                else:
                    avg_scores.append(3)
        elif all_scores:
            avg_scores = [round(statistics.mean(s)) for s in zip(*all_scores)]
        else:
            avg_scores = [3, 3, 3, 3, 3]
        avg_conf = ([round(statistics.mean(c), 3) for c in zip(*all_conf)]
                    if all_conf else [0.5] * 5)
        total = sum(avg_scores)
        overall_conf = round(statistics.mean(avg_conf), 3)
        return {"scores": avg_scores, "confidence": avg_conf, "total_score": total,
                "overall_confidence": overall_conf,
                "consensus_quality": ParallelJudgeSystem._classify_consensus_quality(overall_conf, None),
                "overall_quality": "PASS" if total >= _PASS_THRESHOLD else "FAIL"}

    @staticmethod
    def _calculate_agreement(evaluations: List[Dict]) -> float:
        if len(evaluations) < 2:
            return 1.0
        all_scores = [e["scores"] for e in evaluations if "scores" in e]
        all_conf = [e["confidence"] for e in evaluations if "confidence" in e]
        if len(all_scores) < 2:
            return 0.0
        agreements = []
        for k in range(5):
            scores = [s[k] for s in all_scores]
            confs = ([c[k] for c in all_conf] if all_conf and len(all_conf) == len(all_scores)
                     else [0.5] * len(scores))
            diffs = []
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    diffs.append(abs(scores[i] - scores[j]) * ((confs[i] * confs[j]) ** 0.5))
            if diffs:
                agreements.append(max(0.0, 1.0 - statistics.mean(diffs) / 4.0))
        return round(statistics.mean(agreements), 3) if agreements else 0.0

    @staticmethod
    def _classify_consensus_quality(overall_confidence: float,
                                    agreement_level: Optional[float]) -> str:
        if agreement_level is None:
            if overall_confidence > 0.7:
                return "HIGH"
            return "MEDIUM" if overall_confidence > 0.5 else "LOW"
        if overall_confidence > 0.7 and agreement_level > 0.8:
            return "HIGH"
        if overall_confidence > 0.5 or agreement_level > 0.6:
            return "MEDIUM"
        return "LOW"

    def _generate_summary_report(self, model_results: Dict[str, List[Dict]],
                                 combined_results: List[Dict]):
        report = {"summary": {"total_input_records": len(combined_results),
                              "models_completed": list(model_results.keys()),
                              "successful_combinations": len(combined_results),
                              "deployment_method": "parallel"},
                  "model_stats": {}, "combined_stats": {}}
        for model_name, results in model_results.items():
            valid = [r[f"{model_name}_evaluation"] for r in results
                     if r.get(f"{model_name}_evaluation", {}).get("valid")]
            if valid:
                scores = [e["total_score"] for e in valid]
                report["model_stats"][model_name] = {
                    "total_evaluations": len(valid),
                    "avg_total_score": round(statistics.mean(scores), 2),
                    "avg_confidence": round(statistics.mean(e["avg_confidence"] for e in valid), 3),
                    "pass_rate": sum(1 for s in scores if s >= _PASS_THRESHOLD) / len(scores)}
        if combined_results:
            cs = [r["combined_evaluation"] for r in combined_results]
            scores = [e["total_score"] for e in cs]
            report["combined_stats"] = {
                "avg_total_score": round(statistics.mean(scores), 2),
                "avg_confidence": round(statistics.mean(e["overall_confidence"] for e in cs), 3),
                "avg_agreement": round(statistics.mean(r["agreement_level"] for r in combined_results), 3),
                "pass_rate": sum(1 for s in scores if s >= _PASS_THRESHOLD) / len(scores)}
        with open(self.output_dir / "evaluation_summary_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Summary report saved to %s", self.output_dir / "evaluation_summary_report.json")


async def _amain():
    parser = argparse.ArgumentParser(description="Parallel multi-agent LLM-as-a-judge")
    parser.add_argument("--input", required=True, help="Input JSONL with narratives")
    parser.add_argument("--output-dir", default="./parallel_judge_results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-deploy", action="store_true",
                        help="Assume judge models are already serving (do not launch SGLang)")
    args = parser.parse_args()
    config = ParallelJudgeConfig(input_file=args.input, output_dir=args.output_dir,
                                 batch_size=args.batch_size, max_records=args.max_records,
                                 models_to_run=args.models, temperature=args.temperature,
                                 deploy=not args.no_deploy)
    await ParallelJudgeSystem(config).run_parallel_evaluation()


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
