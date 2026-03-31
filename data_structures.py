# -*- coding: utf-8 -*-
"""
Usage:
    Data structure definition module for all data classes used in the AutoVQA-G framework.
    It includes images, captions, VQA pairs, visual grounding instances, generated data,
    verification outputs, and high-quality data points.
    This module is not meant to be run directly; it is imported by pipeline.py and modules.py.
"""

# Copyright (c) 2025 Rongsheng Hu
# Licensed under the MIT License. See LICENSE in the project root for license information.

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Image:
    """Represents an image, e.g., by its path or ID."""
    identifier: str


@dataclass
class Caption:
    text: str


@dataclass
class VQAPair:
    question: str
    answer: str


@dataclass
class VisualGroundingInstance:
    object_description: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    gn_raw_output: Optional[str] = None


@dataclass
class GeneratedData:
    caption: Caption
    vqa_pair: VQAPair
    visual_grounding: VisualGroundingInstance


@dataclass
class VerificationOutput:
    cot_critique: str
    consistency_score: float  # Between 0.0 and 1.0


@dataclass
class HighQualityDataPoint:
    image_identifier: str
    question: str
    answer: str
    object_description: str
    bbox: Tuple[int, int, int, int]
