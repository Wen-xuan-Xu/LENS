"""Sequential multi-agent LLM-as-a-judge.

Deploys one judge model at a time, evaluates all narratives with it, tears it
down, then moves to the next model.  Slower than :mod:`parallel_judge` but
uses far less GPU memory (only one model resident at a time).  Scoring,
combination, and reporting logic are shared with the parallel implementation.

Heavy / optional dependencies (``openai``, ``aiohttp``) are imported lazily.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .judge_config import JUDGE_MODELS_AVAILABLE
from .parallel_judge import ParallelJudgeSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SequentialJudgeConfig:
    """Configuration for the sequential judge run."""

    input_file: str
    output_dir: str = "./sequential_judge_results"
    batch_size: int = 8
    max_records: int = 0
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 180
    models_to_run: Optional[List[str]] = None
    deploy: bool = True


class SequentialJudgeSystem(ParallelJudgeSystem):
    """Sequential LLM judge system -- one model at a time."""

    def __init__(self, config: SequentialJudgeConfig):
        # Reuse the parallel base for the shared evaluation/combination helpers.
        self.config = config  # type: ignore[assignment]
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if config.models_to_run:
            self.models = [m for m in JUDGE_MODELS_AVAILABLE if m.model_name in config.models_to_run]
        else:
            self.models = JUDGE_MODELS_AVAILABLE.copy()
        logger.info("Models to run sequentially: %s", [m.model_name for m in self.models])

    async def run_sequential_evaluation(self):
        input_data = self._load_input_data()
        if not input_data:
            logger.error("No input data loaded")
            return
        logger.info("Loaded %d records for evaluation", len(input_data))

        model_results: Dict[str, List[Dict]] = {}
        for i, model_config in enumerate(self.models):
            logger.info("Evaluation %d/%d: %s", i + 1, len(self.models), model_config.model_name)
            if self.config.deploy and not await self._deploy_single_model(model_config, model_config.gpu_id):
                logger.error("Failed to deploy %s, skipping", model_config.model_name)
                continue
            try:
                res = await self._evaluate_with_single_model(model_config, input_data)
                model_results[model_config.model_name] = res
                out_file = self.output_dir / f"{model_config.model_name}_evaluations.jsonl"
                self._save_jsonl(res, out_file)
                logger.info("Saved %s results to %s", model_config.model_name, out_file)
            finally:
                if self.config.deploy:
                    await self._cleanup_model(model_config)

        if len(model_results) >= 2:
            combined = self._combine_model_results(input_data, model_results)
            combined_file = self.output_dir / "combined_judge_results.jsonl"
            self._save_jsonl(combined, combined_file)
            logger.info("Combined results saved to %s", combined_file)
            self._generate_summary_report(model_results, combined)
        else:
            logger.warning("Fewer than 2 models completed; no combined results")


async def _amain():
    parser = argparse.ArgumentParser(description="Sequential multi-agent LLM-as-a-judge")
    parser.add_argument("--input", required=True, help="Input JSONL with narratives")
    parser.add_argument("--output-dir", default="./sequential_judge_results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-deploy", action="store_true",
                        help="Assume judge models are already serving (do not launch SGLang)")
    args = parser.parse_args()
    config = SequentialJudgeConfig(input_file=args.input, output_dir=args.output_dir,
                                   batch_size=args.batch_size, max_records=args.max_records,
                                   models_to_run=args.models, temperature=args.temperature,
                                   deploy=not args.no_deploy)
    await SequentialJudgeSystem(config).run_sequential_evaluation()


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
