# -*- coding: utf-8 -*-
"""
Usage:
    Data generation pipeline that orchestrates the iterative
    "generate-verify-optimize" loop of AutoVQA-G.
    This module connects the five core components, CaptionGenerator,
    VQAGenerator, VisualGroundingGenerator, CoTGenerativeVerifier, and
    PromptOptimizer, into a complete self-improving annotation pipeline.

    Main workflow:
      1. Generation stage: call Caption, VQA, and VG modules in sequence to produce candidate annotations.
      2. Verification stage: use the CoT verifier to assess annotation quality and compute a combined consistency score.
      3. Optimization stage: if the score is below the threshold, the prompt optimizer iteratively improves prompts based on feedback.
      Repeat until the score reaches the threshold or the maximum number of iterations is reached.
"""

# Copyright (c) 2025 Rongsheng Hu
# Licensed under the MIT License. See LICENSE in the project root for license information.

import json
import logging
import os
import time
import datetime
from typing import Dict, List

from data_structures import Image, GeneratedData
from modules import (
    CaptionGenerator,
    VQAGenerator,
    VisualGroundingGenerator,
    CoTGenerativeVerifier,
    PromptOptimizer,
)


class DataGenerationPipeline:
    """Core AutoVQA-G pipeline for self-improving VQA-G annotation generation."""

    def __init__(
        self,
        initial_prompts: Dict[str, str],
        consistency_threshold: float = 0.9,
        max_iterations: int = 5,
        output_dir: str = "output",
    ):
        self.caption_generator = CaptionGenerator()
        self.vqa_generator = VQAGenerator()
        self.visual_grounding_generator = VisualGroundingGenerator()
        self.verifier = CoTGenerativeVerifier()
        self.prompt_optimizer = PromptOptimizer()

        self.initial_prompts = initial_prompts.copy()
        self.consistency_threshold = consistency_threshold
        self.max_iterations = max_iterations

        # Save path for prompt optimization history
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.prompt_history_path = os.path.join(output_dir, f"prompt_history_{ts}.json")

        logging.info(
            f"DataGenerationPipeline initialized. "
            f"Threshold={consistency_threshold}, MaxIter={max_iterations}"
        )

    # ------------------------------------------------------------------
    def _save_prompt_history_for_image(self, image_id: str, history: List[Dict]):
        """Append and save the full prompt history for a single image."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entries = []
        for entry in history:
            e = entry.copy()
            e["image_id"] = image_id
            e["timestamp"] = timestamp
            entries.append(e)

        existing = []
        if os.path.exists(self.prompt_history_path):
            with open(self.prompt_history_path, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []

        existing.extend(entries)
        with open(self.prompt_history_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)

    # ------------------------------------------------------------------
    def run(self, image: Image) -> dict:
        """
        Run the full generate-verify-optimize loop for a single image.

        Prompts are reset for each image so optimization history does not leak
        across samples. If no iteration reaches the threshold, the highest-score
        candidate from the current image is returned.
        """
        logging.info(f"Starting pipeline for image: {image.identifier}")
        prompts = self.initial_prompts.copy()
        all_iteration_results: List[dict] = []
        best_result = None
        prompts_history = [
            {"iteration": 0, "prompts": prompts.copy(), "critique": "Initial prompts."}
        ]

        for i in range(self.max_iterations):
            logging.info(f"Iteration {i + 1}/{self.max_iterations}")

            # ---- Generation Phase ----
            caption, usage_cap = self.caption_generator.generate(image, prompts["caption"])
            vqa_pair, usage_vqa = self.vqa_generator.generate(image, caption, prompts["vqa"])
            vg, usage_vg = self.visual_grounding_generator.generate(
                image, caption, vqa_pair, prompts["vg_mention"]
            )
            generated_data = GeneratedData(caption=caption, vqa_pair=vqa_pair, visual_grounding=vg)

            # ---- Consistency Check Phase ----
            verification, usage_ver = self.verifier.verify(
                image, generated_data,
                prompts["content_verification"],
                prompts["bbox_verification"],
            )

            current_output = {
                "image_identifier": image.identifier,
                "question": vqa_pair.question,
                "answer": vqa_pair.answer,
                "object_description": vg.object_description,
                "bbox": vg.bbox,
                "GroundingDINO_raw_output": vg.gn_raw_output,
                "CoT_critique": verification.cot_critique,
                "consistency_score": verification.consistency_score,
                "iteration": i + 1,
                "is_successful": verification.consistency_score >= self.consistency_threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "llm_usage": {
                    "caption": usage_cap,
                    "vqa": usage_vqa,
                    "visual_grounding": usage_vg,
                    "verification": usage_ver,
                },
            }
            all_iteration_results.append(current_output)
            if (
                best_result is None
                or current_output["consistency_score"] > best_result["consistency_score"]
            ):
                best_result = current_output.copy()

            if verification.consistency_score >= self.consistency_threshold:
                logging.info(
                    f"Threshold met ({verification.consistency_score:.2f} >= "
                    f"{self.consistency_threshold}). Stopping."
                )
                break

            # ---- Optimization Phase ----
            if i < self.max_iterations - 1:
                prompts_history[-1]["critique"] = verification.cot_critique
                prompts, usage_opt = self.prompt_optimizer.optimize(
                    verification, prompts,
                    prompts["prompt_optimization"],
                    prompts_history=prompts_history,
                )
                prompts = {
                    k: v for k, v in prompts.items()
                    if k not in ["analysis", "plans_to_improve"]
                }
                prompts_history.append({
                    "iteration": i + 1,
                    "prompts": prompts.copy(),
                    "critique": "",
                    "llm_usage": usage_opt,
                })
            else:
                logging.warning("Max iterations reached without meeting threshold.")

        self._save_prompt_history_for_image(image.identifier, prompts_history)
        if best_result is None:
            logging.warning(f"No iteration results generated for {image.identifier}")
            return {}

        if not best_result["is_successful"]:
            logging.info(
                f"No iteration met threshold for {image.identifier}; "
                f"returning best score {best_result['consistency_score']:.2f} "
                f"from iteration {best_result['iteration']}."
            )

        return best_result
