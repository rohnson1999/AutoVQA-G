# AutoVQA-G: Self-Improving Agentic Framework for Automated Visual Question Answering and Grounding Annotation

<div align="center">

![AutoVQG Framework](img/main_img.jpg)

**A self-improving agentic framework for automated high-fidelity Visual Question Answering & Grounding (VQA-G) dataset generation**

**Accepted at ICASSP 2026 (Poster)**

[![Paper](https://img.shields.io/badge/Paper-ICASSP%202026-blue)](https://github.com/rohnson1999/AutoVQA-G) [![Code](https://img.shields.io/badge/Code-Released-green)](https://github.com/rohnson1999/AutoVQA-G) [![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

## Overview

Manual annotation of high-quality Visual Question Answering with Grounding (VQA-G) datasets, which pair visual questions with evidential grounding, is crucial for advancing vision-language models (VLMs), but remains unscalable. Existing automated methods are often hindered by two key issues: **(1)** inconsistent data fidelity due to model hallucinations; **(2)** brittle verification mechanisms based on simple heuristics. To address these limitations, we introduce **AutoVQA-G**, a self-improving agentic framework for automated VQA-G annotation. AutoVQA-G employs an iterative refinement loop where a **Consistency Evaluation** module uses Chain-of-Thought (CoT) reasoning for fine-grained visual verification. Based on this feedback, a memory-augmented **Prompt Optimization** agent analyzes critiques from failed samples to progressively refine generation prompts. Our experiments show that AutoVQA-G generates VQA-G datasets with superior visual grounding accuracy compared to leading multimodal LLMs, offering a promising approach for creating high-fidelity data to facilitate more robust VLM training and evaluation.

## News

- **[2026/03]** 🎉 Paper accepted at **ICASSP 2026** as a Poster presentation! Full code released.
- **[2025/09]** Initial prompt templates released.

## Project Structure

```
AutoVQA-G/
├── main.py                # Entry point — batch annotation generation
├── pipeline.py            # Core pipeline — generate-evaluate-refine loop
├── modules.py             # All functional modules (generation, verification, optimization)
├── data_structures.py     # Dataclass definitions
├── requirements.txt       # Python dependencies
├── prompts/
│   └── system_prompts.py  # Prompt templates for all modules
├── img/                   # Figures
└── LICENSE
```

## Installation

```bash
git clone https://github.com/rohnson1999/AutoVQA-G.git
cd AutoVQA-G
pip install -r requirements.txt
```

## Configuration

Set these environment variables before running:

| Variable | Purpose | Default endpoint / model |
|---|---|---|
| `GENERATION_API_KEY` | Local generation server for caption, VQA, and mention generation | `MiniCPM-o 2.6` via `http://localhost:8000/v1` |
| `VERIFICATION_API_KEY` | CoT verifier | `Qwen2.5-VL-72B-Instruct` via DashScope |
| `OPTIMIZER_API_KEY` | Prompt optimizer | `DeepSeek V3` via DeepSeek API |
| `GROUNDING_DINO_URL` | GroundingDINO HTTP endpoint | `http://localhost:8001/predict` |
| `GROUNDING_DINO_API_KEY` | Optional auth for GroundingDINO | optional |

```bash
# Linux / macOS
export GENERATION_API_KEY="your_generation_api_key"
export VERIFICATION_API_KEY="your_verification_api_key"
export OPTIMIZER_API_KEY="your_optimizer_api_key"
export GROUNDING_DINO_URL="http://localhost:8001/predict"

# Windows PowerShell
$env:GENERATION_API_KEY="your_generation_api_key"
$env:VERIFICATION_API_KEY="your_verification_api_key"
$env:OPTIMIZER_API_KEY="your_optimizer_api_key"
$env:GROUNDING_DINO_URL="http://localhost:8001/predict"
```

GroundingDINO should accept a `POST` request with `image`, `query`, `box_threshold`, and `text_threshold`, and return a `detections` list containing `bbox` and `confidence`.

> You can change models or endpoints in `modules.py`.

## Usage

### Input Format

Prepare a JSON annotation file as a list of dictionaries, each with at least an `"image"` key:

```json
[
    {"image": "image_001.jpg"},
    {"image": "image_002.jpg"}
]
```

### Running

```bash
python main.py \
    --annotations  path/to/annotations.json \
    --image_dir    path/to/images/ \
    --output_dir   output/ \
    --threshold    0.9 \
    --max_iter     5
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--annotations` | (required) | Path to annotations JSON file |
| `--image_dir` | (required) | Directory containing input images |
| `--output_dir` | `output/` | Directory for results and logs |
| `--threshold` | `0.9` | Consistency score threshold (τ) for acceptance |
| `--max_iter` | `5` | Maximum refinement iterations per image |
| `--start` | `0` | Start index in annotations list |
| `--end` | `None` | End index in annotations list |

### Output Format

Results are saved as a JSON file in `output_dir`. Each image contributes one selected result: the first sample that passes the threshold, or otherwise the highest-scoring iteration for that image.

```json
{
    "image_identifier": "path/to/image.jpg",
    "question": "What color is the car on the left?",
    "answer": "The car on the left is red.",
    "object_description": "red car on left",
    "bbox": [120, 85, 340, 260],
    "consistency_score": 0.92,
    "iteration": 2,
    "is_successful": true,
    "CoT_critique": "...",
    "timestamp": "2025-07-10 14:30:00"
}
```

## Framework Architecture

The AutoVQA-G pipeline operates through an iterative **Generate → Evaluate → Refine** loop:

1. **Modular Generation** (§2.1): Caption Reasoning → VQA Generation → Visual Grounding (Object Mention + GroundingDINO BBox)
2. **CoT Consistency Verification** (§2.2): Two-stage assessment — content quality verification + bbox accuracy verification — producing weighted consistency scores
3. **Memory-Augmented Prompt Optimization** (§2.3): Analyzes verification critiques, maintains iteration history, and dynamically refines generation prompts

The loop terminates when the consistency score `S_t ≥ τ` or the maximum number of iterations is reached. Prompt optimization is reset per image, so one sample's refinement history does not affect the next sample.

## Qualitative Examples

Qualitative examples generated by AutoVQA-G. The framework successfully produces high-consistency data across diverse scenarios, showcasing complex reasoning in QA pairs and precise, fine-grained visual grounding. **(Better viewed zoomed in)**

![Qualitative Examples](img/Qualitative_Results.jpg)

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{hu2026autovqag,
    title={AutoVQA-G: Self-Improving Agentic Framework for Automated Visual Question Answering and Grounding Annotation},
    author={Hu, Rongsheng and Guan, Runwei and Di, Yicheng and Bao, Jiayu and Liu, Yuan},
    booktitle={IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
    year={2026}
}
```

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant No. 62472200).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
