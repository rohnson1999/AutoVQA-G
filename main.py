# -*- coding: utf-8 -*-
"""
Usage:
    Main entry script for AutoVQA-G. It processes images in batches and generates
    VQA-G annotation data.

    Run:
        python main.py \
            --annotations  path/to/annotations.json \
            --image_dir    path/to/images/ \
            --output_dir   output/ \
            --threshold    0.9 \
            --max_iter     5

    Environment variables (must be set in advance):
        GENERATION_API_KEY     API key for the local vLLM server (MiniCPM-o 2.6)
        VERIFICATION_API_KEY   API key for the CoT verifier (Qwen2.5-VL-72B via DashScope)
        OPTIMIZER_API_KEY      API key for the prompt optimizer (DeepSeek V3)
        GROUNDING_DINO_URL     Local HTTP endpoint for the GroundingDINO service
        GROUNDING_DINO_API_KEY Optional API key for the local GroundingDINO service

    Output:
        A JSON file containing one selected result per image will be generated in output_dir.
"""

# Copyright (c) 2025 Rongsheng Hu
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import json
import logging
import argparse
import datetime

from tqdm import tqdm

from data_structures import Image
from pipeline import DataGenerationPipeline
from prompts.system_prompts import initial_prompts_V2


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoVQA-G: Self-Improving Agentic Framework for VQA-G Annotation"
    )
    parser.add_argument(
        "--annotations", type=str, required=True,
        help="Path to the annotations JSON file (list of dicts with 'image' key)."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing the input images."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory for saving results (default: output/)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.9,
        help="Consistency score threshold for acceptance (default: 0.9)."
    )
    parser.add_argument(
        "--max_iter", type=int, default=5,
        help="Maximum refinement iterations per image (default: 5)."
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start index in the annotations list (default: 0)."
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="End index in the annotations list (default: all)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create the output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=os.path.join(log_dir, f"{timestamp}.log"),
    )

    # Load annotation data
    with open(args.annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)[args.start : args.end]

    logging.info(f"Loaded {len(annotations)} annotations.")

    # Initialize the pipeline
    pipeline = DataGenerationPipeline(
        initial_prompts=initial_prompts_V2,
        consistency_threshold=args.threshold,
        max_iterations=args.max_iter,
        output_dir=args.output_dir,
    )

    results_path = os.path.join(args.output_dir, f"annotations_result_{timestamp}.json")

    all_results = []
    for ann in tqdm(annotations, desc="Processing Images", unit="image"):
        img_path = os.path.join(args.image_dir, ann["image"])
        sample_image = Image(identifier=img_path)

        best_result = pipeline.run(sample_image)

        if best_result:
            all_results.append(best_result)
            if best_result.get("question"):
                logging.info(
                    f"Generated data for {img_path}: "
                    f"Q={best_result['question']} (iter={best_result['iteration']}, "
                    f"score={best_result['consistency_score']:.2f})"
                )
            else:
                logging.warning(f"No valid data for {img_path} in selected result.")
        else:
            logging.warning(f"Failed to generate data for {img_path}")

        # Save immediately after each image is processed
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

    logging.info(f"Done. {len(all_results)} total results saved to {results_path}")


if __name__ == "__main__":
    main()
