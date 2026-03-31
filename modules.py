# -*- coding: utf-8 -*-
"""
Usage:
    Core modules implementing each stage of the AutoVQA-G framework.
    Includes:
      - CaptionGenerator: image caption generation
      - VQAGenerator: visual question-answer pair generation
      - VisualGroundingGenerator: visual grounding generation (object mention + bbox)
      - CoTGenerativeVerifier: consistency verification based on chain-of-thought reasoning
      - PromptOptimizer: memory-enhanced prompt optimization

    Configuration:
      1. API keys should be provided through environment variables:
         - GENERATION_API_KEY: API key for the local vLLM server (MiniCPM-o 2.6)
         - VERIFICATION_API_KEY: API key for the CoT verifier (Qwen2.5-VL-72B via DashScope)
         - OPTIMIZER_API_KEY: API key for the prompt optimizer (DeepSeek V3)
         - GROUNDING_DINO_URL: Local HTTP endpoint for the GroundingDINO inference service
         - GROUNDING_DINO_API_KEY: Optional API key for the local GroundingDINO service
      2. Model names and API endpoints can be customized via module constructor parameters.
"""

# Copyright (c) 2025 Rongsheng Hu
# Licensed under the MIT License. See LICENSE in the project root for license information.

import logging
import os
import base64
import time
import json
import re
import io
from urllib import error, request

from openai import OpenAI
from typing import Dict, List, Optional
from PIL import Image as PILImage
from PIL import ImageDraw

from data_structures import (
    Image, Caption, VQAPair, VisualGroundingInstance,
    GeneratedData, VerificationOutput
)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def create_bbox_overlay_image(image_path: str, bbox: tuple) -> str:
    """
    Overlay a red bounding box on the original image and return the processed image
    as a base64 string.
    """
    try:
        image = PILImage.open(image_path)
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        logging.error(f"Failed to create bbox overlay image: {e}")
        return image_to_base64(image_path)


def image_to_base64(image_path: str) -> str:
    """Encode an image file as a data-URI base64 string."""
    with open(image_path, "rb") as img_f:
        encoded = base64.b64encode(img_f.read()).decode("utf-8")
    ext = image_path.split('.')[-1].lower()
    mime = "jpeg" if ext in ["jpg", "jpeg"] else "png" if ext == "png" else "octet-stream"
    return f"data:image/{mime};base64,{encoded}"


def api_call_with_retry(
    messages,
    model: str = "openbmb/MiniCPM-o-2_6",
    max_retries: int = 3,
    delay: float = 1.0,
    api_key: Optional[str] = None,
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 1.0,
):
    """
    Generic API call helper with exponential backoff retries.
    By default it connects to a local vLLM inference service (MiniCPM-o 2.6),
    while remaining compatible with any OpenAI-style API.

    Args:
        messages: A list of messages in OpenAI format.
        model: Model name, defaulting to MiniCPM-o 2.6 as the generation model used in the paper.
        max_retries: Maximum number of retry attempts.
        delay: Initial retry interval in seconds.
        api_key: API key. If None, reads from the GENERATION_API_KEY environment variable.
        base_url: Base API URL, which defaults to the local vLLM service.
        temperature: Sampling temperature.
    """
    if api_key is None:
        api_key = os.environ.get("GENERATION_API_KEY", "")

    client = OpenAI(base_url=base_url, api_key=api_key)

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            if completion and completion.choices:
                usage = getattr(completion, "usage", None)
                usage_dict = usage.model_dump() if usage and hasattr(usage, "model_dump") else {}
                return completion, usage_dict
            logging.warning(
                f"API call attempt {attempt + 1} returned invalid response. Retrying..."
            )
        except Exception as e:
            logging.warning(f"API call attempt {attempt + 1} failed: {e}")

        if attempt < max_retries - 1:
            time.sleep(delay * (2 ** attempt))

    logging.error(f"API call failed after {max_retries} attempts. Returning placeholder.")

    class _Msg:
        content = "Error: API call failed after multiple retries."

    class _Choice:
        message = _Msg()

    class _Comp:
        choices = [_Choice()]

    return _Comp(), {}


def extract_xml_content(text: str, tag: str) -> str:
    """Extract the content of a specific XML tag from text."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Base Module
# ---------------------------------------------------------------------------

class BaseModule:
    def __init__(self, module_name: str):
        self.module_name = module_name
        logging.info(f"{self.module_name} initialized.")


# ---------------------------------------------------------------------------
# Generation Modules
# ---------------------------------------------------------------------------

class CaptionGenerator(BaseModule):
    """Image caption generation module (corresponding to Eq. 1 in the paper)."""

    def __init__(self):
        super().__init__("CaptionGenerator")

    def generate(self, image: Image, prompt: str):
        logging.info(f"[{self.module_name}] Generating caption for {image.identifier}")

        img_b64 = image_to_base64(image.identifier)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_b64}},
                ],
            }
        ]

        completion, usage = api_call_with_retry(messages)
        content = completion.choices[0].message.content
        logging.info(f"Caption: {content}")
        return Caption(text=content), usage


class VQAGenerator(BaseModule):
    """Visual question-answer pair generation module (Eq. 2 in the paper)."""

    def __init__(self):
        super().__init__("VQAGenerator")

    def generate(self, image: Image, caption: Caption, prompt: str):
        logging.info(f"[{self.module_name}] Generating VQA pair for {image.identifier}")

        img_b64 = image_to_base64(image.identifier)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\nCaption: {caption.text}"},
                    {"type": "image_url", "image_url": {"url": img_b64}},
                ],
            }
        ]

        completion, usage = api_call_with_retry(messages)
        content = completion.choices[0].message.content
        question = extract_xml_content(content, "question")
        answer = extract_xml_content(content, "answer")

        # Fallback parsing
        if question == content:
            if "Q:" in content and "A:" in content:
                question = content.split("Q:")[1].split("A:")[0].strip()
                answer = content.split("A:")[1].strip()
            else:
                question, answer = content, ""

        logging.info(f"VQA: Q={question}, A={answer}")
        return VQAPair(question=question, answer=answer), usage


class VisualGroundingGenerator(BaseModule):
    """
    Visual grounding generation module (corresponding to Eq. 3-4 in the paper).
    Two-step process: 1) generate the object mention; 2) use GroundingDINO to produce the bbox.
    """

    def __init__(self):
        super().__init__("VisualGroundingGenerator")

    def generate_object_mention(self, image: Image, caption: Caption, vqa_pair: VQAPair, prompt: str):
        """Step 1: generate a RefCOCO-style object mention."""
        logging.info(f"[{self.module_name}] Generating object mention for {image.identifier}")

        img_b64 = image_to_base64(image.identifier)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}\n"
                            f"Caption: {caption.text}\n"
                            f"Q: {vqa_pair.question}\n"
                            f"A: {vqa_pair.answer}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": img_b64}},
                ],
            }
        ]

        completion, usage = api_call_with_retry(messages)
        content = completion.choices[0].message.content
        object_mention = extract_xml_content(content, "mention")

        if object_mention == content:
            if "Mention:" in content:
                object_mention = content.split("Mention:")[1].strip()
            else:
                object_mention = content.strip()

        # Clean the mention: keep only English letters, numbers, spaces, and periods
        def _clean(mention: str) -> str:
            cleaned = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', mention)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip().strip('.')
            return cleaned

        original = object_mention
        object_mention = _clean(object_mention)

        if len(object_mention.strip()) < 2:
            logging.warning(f"Object mention too short after cleaning: '{original}' -> '{object_mention}'")
            object_mention = "main object"

        if not re.match(r'^[a-zA-Z0-9\s\.]+$', object_mention):
            object_mention = re.sub(r'[^a-zA-Z0-9\s\.]', '', object_mention).strip()

        if original != object_mention:
            logging.info(f"Object mention cleaned: '{original}' -> '{object_mention}'")

        logging.info(f"VG Object Mention: {object_mention}")
        return object_mention, usage

    def _parse_grounding_dino_output(self, output: dict, object_mention: str) -> tuple:
        """Parse GroundingDINO output and return the bbox with the highest confidence."""
        try:
            detections = output.get('detections', [])
            if not detections:
                logging.warning(f"No detections found for '{object_mention}'")
                return (0, 0, 0, 0)

            best = max(detections, key=lambda x: x.get('confidence', 0))
            bbox = best.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            logging.info(
                f"Best detection for '{object_mention}': "
                f"bbox=({x1},{y1},{x2},{y2}), conf={best.get('confidence', 0):.3f}"
            )
            return (x1, y1, x2, y2)
        except Exception as e:
            logging.error(f"Failed to parse GroundingDINO output: {e}")
            return (0, 0, 0, 0)

    def generate_bbox_groundingdino(self, image: Image, object_mention: str) -> tuple:
        """
        Generate a bbox with a locally deployed GroundingDINO service.

        The service is expected to expose an HTTP endpoint that accepts JSON like:
        {
            "image": "<data-uri-base64>",
            "query": "<object mention>",
            "box_threshold": 0.2,
            "text_threshold": 0.2
        }
        and returns a JSON object containing a "detections" list.
        """
        logging.info(
            f"[{self.module_name}] Generating bbox via local GroundingDINO for '{object_mention}'"
        )

        endpoint = os.environ.get("GROUNDING_DINO_URL", "http://localhost:8001/predict")
        api_key = os.environ.get("GROUNDING_DINO_API_KEY", "")
        max_retries = 3

        for attempt in range(max_retries):
            try:
                payload = {
                    "image": image_to_base64(image.identifier),
                    "query": object_mention,
                    "box_threshold": 0.2,
                    "text_threshold": 0.2,
                }
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                req = request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with request.urlopen(req, timeout=60) as resp:
                    output = json.loads(resp.read().decode("utf-8"))

                bbox = self._parse_grounding_dino_output(output, object_mention)
                logging.info(f"VG Bbox (GroundingDINO): {bbox}")
                return bbox, str(output)

            except error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                logging.warning(
                    f"GroundingDINO attempt {attempt + 1}/{max_retries} failed with "
                    f"HTTP {e.code}: {body}"
                )
            except Exception as e:
                logging.warning(
                    f"GroundingDINO attempt {attempt + 1}/{max_retries} failed: {e}"
                )

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        logging.error(
            f"Local GroundingDINO inference failed after {max_retries} attempts. "
            f"Endpoint: {endpoint}"
        )
        return (0, 0, 0, 0), None

    def generate(self, image: Image, caption: Caption, vqa_pair: VQAPair, mention_prompt: str):
        """Combined method that runs the full two-step visual grounding pipeline."""
        logging.info(f"[{self.module_name}] Starting two-step VG for {image.identifier}")

        object_mention, usage_mention = self.generate_object_mention(
            image, caption, vqa_pair, mention_prompt
        )
        bbox, gn_raw_output = self.generate_bbox_groundingdino(image, object_mention)

        return (
            VisualGroundingInstance(
                object_description=object_mention,
                bbox=bbox,
                gn_raw_output=gn_raw_output,
            ),
            usage_mention,
        )


# ---------------------------------------------------------------------------
# Verification Module
# ---------------------------------------------------------------------------

class CoTGenerativeVerifier(BaseModule):
    """
    Consistency verification module based on chain-of-thought reasoning
    (corresponding to Section 2.2 of the paper).
    Two-stage verification: 1) content quality; 2) bbox accuracy.
    """

    def __init__(self, verifier_model: str = "qwen2.5-vl-72b-instruct",
                 verifier_api_key: Optional[str] = None,
                 verifier_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__("CoTGenerativeVerifier")
        self.verifier_model = verifier_model
        self.verifier_api_key = verifier_api_key or os.environ.get("VERIFICATION_API_KEY", "")
        self.verifier_base_url = verifier_base_url

    def _extract_step_scores(self, content: str) -> float:
        """Extract step scores from the verifier output and compute their average."""
        step_scores = []
        matches = re.findall(r"<step\d+_score>(.*?)</step\d+_score>", content, re.IGNORECASE)
        for score_str in matches:
            try:
                step_scores.append(float(score_str.strip()))
            except Exception:
                step_scores.append(0.0)

        if step_scores:
            return sum(step_scores) / len(step_scores)

        # Fallback
        if "CoT:" in content and "Score:" in content:
            try:
                return float(content.split("Score:")[1].strip())
            except Exception:
                pass
        return 0.0

    def verify_content_quality(self, image: Image, generated_data: GeneratedData, prompt: str):
        """Stage 1: verify content quality."""
        logging.info(f"[{self.module_name}] Content quality verification for {image.identifier}")

        img_b64 = image_to_base64(image.identifier)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}\n"
                            f"Caption: {generated_data.caption.text}\n"
                            f"Q: {generated_data.vqa_pair.question}\n"
                            f"A: {generated_data.vqa_pair.answer}\n"
                            f"Object Description: {generated_data.visual_grounding.object_description}\n"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": img_b64}},
                ],
            }
        ]

        completion, usage = api_call_with_retry(
            messages, model=self.verifier_model,
            api_key=self.verifier_api_key, base_url=self.verifier_base_url,
        )
        content = completion.choices[0].message.content
        cot = extract_xml_content(content, "cot")
        score = self._extract_step_scores(content)

        logging.info(f"Content Quality Score: {score}")
        return VerificationOutput(cot_critique=cot, consistency_score=score), usage

    def verify_bbox_accuracy(self, image: Image, generated_data: GeneratedData, prompt: str):
        """Stage 2: verify bbox accuracy through image-text consistency evaluation."""
        logging.info(f"[{self.module_name}] Bbox accuracy verification for {image.identifier}")

        bbox_img_b64 = create_bbox_overlay_image(
            image.identifier, generated_data.visual_grounding.bbox
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}\n"
                            f"Object Description: {generated_data.visual_grounding.object_description}\n"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": bbox_img_b64}},
                ],
            }
        ]

        completion, usage = api_call_with_retry(
            messages, model=self.verifier_model,
            api_key=self.verifier_api_key, base_url=self.verifier_base_url,
        )
        content = completion.choices[0].message.content
        cot = extract_xml_content(content, "cot")
        score = self._extract_step_scores(content)

        logging.info(f"Bbox Accuracy Score: {score}")
        return VerificationOutput(cot_critique=cot, consistency_score=score), usage

    def verify(self, image: Image, generated_data: GeneratedData,
               content_prompt: str, bbox_prompt: str,
               content_weight: float = 0.7, bbox_weight: float = 0.3):
        """
        Run the full verification flow by combining both stages with weighted scoring.
        Default weights: w_vqa=0.7 and w_vg=0.3, following the paper.
        """
        logging.info(f"[{self.module_name}] Starting verification for {image.identifier}")

        content_v, usage_c = self.verify_content_quality(image, generated_data, content_prompt)
        bbox_v, usage_b = self.verify_bbox_accuracy(image, generated_data, bbox_prompt)

        combined_cot = (
            f"Content Quality Analysis:\n{content_v.cot_critique}\n\n"
            f"Bbox Accuracy Analysis:\n{bbox_v.cot_critique}"
        )
        combined_score = (
            content_v.consistency_score * content_weight
            + bbox_v.consistency_score * bbox_weight
        )

        logging.info(f"Combined Verification Score: {combined_score:.2f}")
        return (
            VerificationOutput(cot_critique=combined_cot, consistency_score=combined_score),
            {"content_verification": usage_c, "bbox_verification": usage_b},
        )


# ---------------------------------------------------------------------------
# Prompt Optimization Module
# ---------------------------------------------------------------------------

class PromptOptimizer(BaseModule):
    """
    Memory-enhanced prompt optimization module (corresponding to Section 2.3 of the paper).
    It analyzes verification feedback, identifies failure patterns, and iteratively
    improves generation prompts.
    """

    def __init__(self, optimizer_model: str = "deepseek-chat",
                 optimizer_api_key: Optional[str] = None,
                 optimizer_base_url: str = "https://api.deepseek.com"):
        super().__init__("PromptOptimizer")
        self.optimizer_model = optimizer_model
        self.optimizer_api_key = optimizer_api_key or os.environ.get("OPTIMIZER_API_KEY", "")
        self.optimizer_base_url = optimizer_base_url

    def _parse_json_response(self, response_text: str, fallback_data: Optional[Dict] = None) -> Dict:
        """Parse JSON content from an API response with tolerance for multiple formats."""
        if not response_text:
            return fallback_data or {}

        text = response_text.strip()

        # 1. Direct JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Markdown code blocks
        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if cleaned.startswith('{') and cleaned.endswith('}'):
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        continue

        # 3. JSON object pattern
        matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # 4. Find outermost braces
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            json_part = text[start:end]
            # Fix common issues
            json_part = json_part.replace("'", '"')
            json_part = re.sub(r',\s*}', '}', json_part)
            json_part = re.sub(r',\s*]', ']', json_part)
            try:
                return json.loads(json_part)
            except json.JSONDecodeError:
                pass

        logging.error(f"Failed to parse JSON from response: {text[:200]}...")
        return fallback_data or {}

    def optimize(self, verification_output, current_prompts: Dict[str, str],
                 prompt: str, prompts_history: Optional[List[Dict]] = None):
        """Optimize prompts based on verification feedback."""
        logging.info(f"[{self.module_name}] Optimizing prompts based on critique")

        history_str = ""
        if prompts_history:
            history_str = "\n\n--- Prompt Iteration History ---\n"
            for item in prompts_history:
                history_str += f"\n--- Iteration {item['iteration']} ---\n"
                history_str += f"Critique: {item.get('critique', 'N/A')}\n"
                history_str += f"Prompts: {json.dumps(item['prompts'], indent=2)}\n"
            history_str += "--------------------------------\n\n"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}\n\n"
                            f"--- Current Iteration's Critique ---\n"
                            f"{verification_output.cot_critique}\n\n"
                            f"--- Current Prompts ---\n"
                            f"{json.dumps(current_prompts, indent=2)}"
                            f"{history_str}"
                        ),
                    },
                ],
            }
        ]

        completion, usage = api_call_with_retry(
            messages, model=self.optimizer_model,
            api_key=self.optimizer_api_key, base_url=self.optimizer_base_url,
            temperature=0.5,
        )
        content = completion.choices[0].message.content

        optimized = self._parse_json_response(content, {})

        if optimized:
            new_prompts = current_prompts.copy()
            xml_tag_pattern = re.compile(r"<[^>]+>")
            key_prompts = ["vqa", "visual_grounding", "verification"]

            valid_updates = {}
            for key, value in optimized.items():
                if key in key_prompts:
                    if xml_tag_pattern.search(value):
                        valid_updates[key] = value
                    else:
                        logging.warning(f"Optimized prompt '{key}' lacks XML tags, keeping original.")
                else:
                    valid_updates[key] = value

            new_prompts.update(valid_updates)

            if valid_updates:
                logging.info("Prompts optimized successfully.")
                return new_prompts, usage

        logging.warning("Failed to parse optimized prompts, using originals.")
        return current_prompts, usage
