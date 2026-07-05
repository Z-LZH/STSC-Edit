#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selection_baselines.py

统一运行多物体目标选择实验中的 baseline：

1. Spatial rule only
2. DINO full phrase
3. DINO category top-1
4. DINO + CLIP ranking
5. OWL-ViT
6. OWLv2
7. KOSMOS-2 grounding

输入：
    image + target_phrase + category

输出：
    JSON / JSONL，每个 method 输出一个 bbox：
    [x1, y1, x2, y2]

推荐用于论文实验：
    先跑 bbox selection，再用 bbox 作为 SAM2 box prompt 得到 mask。

作者注：
    - Spatial rule only 默认使用 DINO category 检测所有同类候选，然后只用空间规则选一个。
    - DINO + CLIP ranking 默认使用 DINO category 产生候选框，再用 CLIP 对 crop 和 target_phrase 排序。
    - OWLv2 / OWL-ViT 默认直接用 target_phrase 检测并取 top-1。
    - KOSMOS-2 默认使用 <grounding><phrase> target_phrase </phrase> 进行 referring expression grounding。
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from PIL import Image, ImageDraw, ImageFont

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    CLIPProcessor,
    CLIPModel,
)

# OWL 类不一定在旧 transformers 中存在，因此延迟导入
try:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
except Exception:
    Owlv2Processor = None
    Owlv2ForObjectDetection = None

try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
except Exception:
    OwlViTProcessor = None
    OwlViTForObjectDetection = None

try:
    from transformers import Kosmos2ForConditionalGeneration
except Exception:
    Kosmos2ForConditionalGeneration = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None


# -------------------------
# 数据结构
# -------------------------

@dataclass
class SelectionResult:
    method: str
    success: bool
    box: Optional[List[float]]
    score: Optional[float]
    label: Optional[str]
    image: Optional[str] = None
    sample_id: Optional[str] = None
    target_phrase: Optional[str] = None
    category: Optional[str] = None
    iou_with_gt: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


# -------------------------
# 通用工具
# -------------------------

def get_device(device_arg: Optional[str] = None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def move_to_device(batch, device: str):
    return {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in batch.items()
    }


def normalize_text_query(text: str, add_photo_prefix: bool = False) -> str:
    text = text.strip()
    if add_photo_prefix:
        if not text.lower().startswith("a photo of"):
            text = "a photo of " + text
    return text


def clamp_box(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))

    if x2 <= x1:
        x2 = min(width - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(height - 1.0, y1 + 1.0)

    return [x1, y1, x2, y2]


def box_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(box_a) + box_area(box_b) - inter

    if union <= 0:
        return 0.0
    return inter / union


def crop_with_padding(
    image: Image.Image,
    box: List[float],
    padding_ratio: float = 0.05,
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    padded_box = [
        x1 - pad_x,
        y1 - pad_y,
        x2 + pad_x,
        y2 + pad_y,
    ]
    x1, y1, x2, y2 = clamp_box(padded_box, width, height)
    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def safe_score(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def serialize_result(result: SelectionResult) -> Dict[str, Any]:
    return asdict(result)


def select_top1_from_detections(
    detections: List[Dict[str, Any]],
    method: str,
    image_path: str,
    sample_id: Optional[str],
    target_phrase: str,
    category: str,
    gt_box: Optional[List[float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> SelectionResult:
    if len(detections) == 0:
        return SelectionResult(
            method=method,
            success=False,
            box=None,
            score=None,
            label=None,
            image=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            iou_with_gt=None,
            extra={"message": "No detection found.", **(extra or {})},
        )

    det = sorted(detections, key=lambda d: d.get("score", 0.0), reverse=True)[0]
    pred_box = det["box"]
    iou = box_iou(pred_box, gt_box) if gt_box is not None else None

    return SelectionResult(
        method=method,
        success=True,
        box=pred_box,
        score=safe_score(det.get("score")),
        label=str(det.get("label")),
        image=image_path,
        sample_id=sample_id,
        target_phrase=target_phrase,
        category=category,
        iou_with_gt=iou,
        extra={
            "num_candidates": len(detections),
            "all_candidates": detections,
            **(extra or {}),
        },
    )


def draw_result(image_path: str, result: SelectionResult, save_path: str):
    image = load_image(image_path)
    draw = ImageDraw.Draw(image)

    if result.success and result.box is not None:
        x1, y1, x2, y2 = result.box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        text = f"{result.method}: {result.label or ''}"
        draw.text((x1, max(0, y1 - 18)), text, fill="red")

    image.save(save_path)


# -------------------------
# 空间规则 baseline
# -------------------------

SPATIAL_KEYWORDS = {
    "left": ["left", "leftmost"],
    "right": ["right", "rightmost"],
    "top": ["top", "upper", "up"],
    "bottom": ["bottom", "lower", "down"],
    "center": ["center", "middle", "central"],
}


def infer_spatial_terms(text: str) -> List[str]:
    text_l = text.lower()
    found = []
    for key, words in SPATIAL_KEYWORDS.items():
        for w in words:
            pattern = r"\b" + re.escape(w) + r"\b"
            if re.search(pattern, text_l):
                found.append(key)
                break
    return found


def spatial_rule_score(
    box: List[float],
    image_width: int,
    image_height: int,
    spatial_terms: List[str],
) -> float:
    """
    分数越高越应该被选中。
    """
    cx, cy = box_center(box)
    nx = cx / max(1.0, image_width)
    ny = cy / max(1.0, image_height)

    if not spatial_terms:
        # 没有空间词时，退化为越居中越好。
        spatial_terms = ["center"]

    scores = []
    for term in spatial_terms:
        if term == "left":
            scores.append(1.0 - nx)
        elif term == "right":
            scores.append(nx)
        elif term == "top":
            scores.append(1.0 - ny)
        elif term == "bottom":
            scores.append(ny)
        elif term == "center":
            dx = abs(nx - 0.5)
            dy = abs(ny - 0.5)
            scores.append(1.0 - (dx + dy))
        else:
            scores.append(0.0)

    return float(sum(scores) / max(1, len(scores)))


# -------------------------
# Grounding DINO wrapper
# -------------------------

class GroundingDINODetector:
    def __init__(
        self,
        model_id: str,
        device: str,
    ):
        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        text_query: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        text_query = normalize_text_query(text_query)

        # GroundingDINO 的 text 输入应该是字符串，不是 [[text_query]]
        # 加句号可以让 GroundingDINO 更稳定地解析 phrase
        text_prompt = text_query.strip()
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = move_to_device(inputs, self.device)

        outputs = self.model(**inputs)

        # Grounding DINO 的官方示例使用 post_process_grounded_object_detection。
        # 不同 transformers 版本里参数名略有差异，因此这里做了兼容。
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )

        result = results[0]
        detections = []

        for box, score, label in zip(
            result.get("boxes", []),
            result.get("scores", []),
            result.get("labels", []),
        ):
            detections.append({
                "box": [float(v) for v in box.detach().cpu().tolist()],
                "score": float(score.detach().cpu().item()),
                "label": str(label),
                "source": "grounding_dino",
            })

        detections = sorted(detections, key=lambda d: d["score"], reverse=True)
        return detections


# -------------------------
# CLIP crop-text ranker
# -------------------------

class CLIPCropRanker:
    def __init__(
        self,
        model_id: str,
        device: str,
    ):
        self.model_id = model_id
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.inference_mode()
    def rank(
        self,
        image: Image.Image,
        boxes: List[List[float]],
        text_query: str,
        padding_ratio: float = 0.05,
    ) -> List[Tuple[int, float]]:
        if len(boxes) == 0:
            return []

        crops = [
            crop_with_padding(image, box, padding_ratio=padding_ratio)
            for box in boxes
        ]

        inputs = self.processor(
            text=[text_query],
            images=crops,
            return_tensors="pt",
            padding=True,
        )
        inputs = move_to_device(inputs, self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image.squeeze(-1)
        probs_or_scores = logits_per_image.detach().cpu().tolist()

        ranked = [(idx, float(score)) for idx, score in enumerate(probs_or_scores)]
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        return ranked


# -------------------------
# OWL-ViT / OWLv2 wrapper
# -------------------------

class OWLDetector:
    def __init__(
        self,
        kind: str,
        model_id: str,
        device: str,
    ):
        assert kind in ["owlvit", "owlv2"]
        self.kind = kind
        self.model_id = model_id
        self.device = device

        if kind == "owlv2":
            if Owlv2Processor is None or Owlv2ForObjectDetection is None:
                raise ImportError(
                    "当前 transformers 版本没有 Owlv2Processor / Owlv2ForObjectDetection。"
                    "请升级：pip install -U transformers"
                )
            self.processor = Owlv2Processor.from_pretrained(model_id)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)

        else:
            if OwlViTProcessor is None or OwlViTForObjectDetection is None:
                raise ImportError(
                    "当前 transformers 版本没有 OwlViTProcessor / OwlViTForObjectDetection。"
                    "请升级：pip install -U transformers"
                )
            self.processor = OwlViTProcessor.from_pretrained(model_id)
            self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)

        self.model.eval()

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        text_query: str,
        threshold: float = 0.10,
        add_photo_prefix: bool = True,
    ) -> List[Dict[str, Any]]:
        text_query = normalize_text_query(text_query, add_photo_prefix=add_photo_prefix)
        text_labels = [[text_query]]

        inputs = self.processor(text=text_labels, images=image, return_tensors="pt")
        inputs = move_to_device(inputs, self.device)

        outputs = self.model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)])

        # OWLv2 新接口一般有 post_process_grounded_object_detection。
        # OWL-ViT 旧接口可能只有 post_process_object_detection。
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
                text_labels=text_labels,
            )
            result = results[0]
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            labels = result.get("text_labels", [text_query] * len(boxes))
        except Exception:
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )
            result = results[0]
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            label_ids = result.get("labels", [])
            labels = []
            for label_id in label_ids:
                try:
                    labels.append(text_labels[0][int(label_id)])
                except Exception:
                    labels.append(text_query)

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detections.append({
                "box": [float(v) for v in box.detach().cpu().tolist()],
                "score": float(score.detach().cpu().item()),
                "label": str(label),
                "source": self.kind,
            })

        detections = sorted(detections, key=lambda d: d["score"], reverse=True)
        return detections


# -------------------------
# KOSMOS-2 wrapper
# -------------------------

class Kosmos2Grounder:
    def __init__(
        self,
        model_id: str,
        device: str,
        torch_dtype: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)

        model_cls = Kosmos2ForConditionalGeneration
        if model_cls is None:
            model_cls = AutoModelForVision2Seq
        if model_cls is None:
            raise ImportError(
                "当前 transformers 版本没有 Kosmos2ForConditionalGeneration 或 AutoModelForVision2Seq。"
                "请升级：pip install -U transformers"
            )

        kwargs = {}
        if torch_dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif torch_dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif torch_dtype == "float32":
            kwargs["torch_dtype"] = torch.float32

        self.model = model_cls.from_pretrained(model_id, **kwargs).to(device)
        self.model.eval()

    @torch.inference_mode()
    def ground(
        self,
        image: Image.Image,
        target_phrase: str,
        category: str,
        max_new_tokens: int = 64,
    ) -> List[Dict[str, Any]]:
        """
        KOSMOS-2 的 referring expression comprehension prompt:
            <grounding><phrase> a snowman next to a fire</phrase>

        返回 entities 中的 bbox。bbox 是归一化坐标，需要转成原图像素坐标。
        """
        prompt = f"<grounding><phrase> {target_phrase.strip()}</phrase>"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = move_to_device(inputs, self.device)

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        processed_text, entities = self.processor.post_process_generation(generated_text)

        width, height = image.size

        detections = []
        for entity_name, span, bboxes in entities:
            for bbox_norm in bboxes:
                x1n, y1n, x2n, y2n = [float(v) for v in bbox_norm]
                box = [
                    x1n * width,
                    y1n * height,
                    x2n * width,
                    y2n * height,
                ]
                box = clamp_box(box, width, height)

                # KOSMOS-2 没有标准 detection confidence，这里设置为 None。
                detections.append({
                    "box": box,
                    "score": 1.0,
                    "label": str(entity_name),
                    "source": "kosmos2",
                    "span": span,
                })

        # 优先选择 entity 文本包含 target/category 的候选；
        # 如果没有，就保留原顺序。
        target_l = target_phrase.lower()
        category_l = category.lower()

        def match_score(det: Dict[str, Any]) -> float:
            label_l = str(det.get("label", "")).lower()
            s = 0.0
            if target_l and target_l in label_l:
                s += 2.0
            if category_l and category_l in label_l:
                s += 1.0
            return s

        detections = sorted(detections, key=match_score, reverse=True)

        for det in detections:
            det["generated_text"] = processed_text

        return detections


# -------------------------
# Runner
# -------------------------

class BaselineRunner:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.device)

        self._dino = None
        self._clip = None
        self._owlv2 = None
        self._owlvit = None
        self._kosmos2 = None

    def get_dino(self) -> GroundingDINODetector:
        if self._dino is None:
            self._dino = GroundingDINODetector(
                model_id=self.args.dino_model,
                device=self.device,
            )
        return self._dino

    def get_clip(self) -> CLIPCropRanker:
        if self._clip is None:
            self._clip = CLIPCropRanker(
                model_id=self.args.clip_model,
                device=self.device,
            )
        return self._clip

    def get_owlv2(self) -> OWLDetector:
        if self._owlv2 is None:
            self._owlv2 = OWLDetector(
                kind="owlv2",
                model_id=self.args.owlv2_model,
                device=self.device,
            )
        return self._owlv2

    def get_owlvit(self) -> OWLDetector:
        if self._owlvit is None:
            self._owlvit = OWLDetector(
                kind="owlvit",
                model_id=self.args.owlvit_model,
                device=self.device,
            )
        return self._owlvit

    def get_kosmos2(self) -> Kosmos2Grounder:
        if self._kosmos2 is None:
            self._kosmos2 = Kosmos2Grounder(
                model_id=self.args.kosmos2_model,
                device=self.device,
                torch_dtype=self.args.kosmos2_dtype,
            )
        return self._kosmos2

    def run_spatial_rule_only(
        self,
        image_path: str,
        target_phrase: str,
        category: str,
        sample_id: Optional[str] = None,
        gt_box: Optional[List[float]] = None,
    ) -> SelectionResult:
        image = load_image(image_path)
        width, height = image.size

        # 使用基础类别产生候选，不使用属性/关系/CLIP，只使用空间规则排序。
        detections = self.get_dino().detect(
            image=image,
            text_query=category,
            box_threshold=self.args.dino_box_threshold,
            text_threshold=self.args.dino_text_threshold,
        )

        spatial_terms = infer_spatial_terms(target_phrase)

        if len(detections) == 0:
            return SelectionResult(
                method="Spatial rule only",
                success=False,
                box=None,
                score=None,
                label=None,
                image=image_path,
                sample_id=sample_id,
                target_phrase=target_phrase,
                category=category,
                extra={
                    "message": "No category candidates from DINO.",
                    "spatial_terms": spatial_terms,
                },
            )

        reranked = []
        for det in detections:
            s = spatial_rule_score(det["box"], width, height, spatial_terms)
            d = dict(det)
            d["spatial_score"] = s
            d["score"] = s
            reranked.append(d)

        reranked = sorted(reranked, key=lambda d: d["spatial_score"], reverse=True)

        return select_top1_from_detections(
            detections=reranked,
            method="Spatial rule only",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
            extra={
                "spatial_terms": spatial_terms,
                "candidate_detector": "DINO category",
            },
        )

    def run_dino_full_phrase(
        self,
        image_path: str,
        target_phrase: str,
        category: str,
        sample_id: Optional[str] = None,
        gt_box: Optional[List[float]] = None,
    ) -> SelectionResult:
        image = load_image(image_path)

        detections = self.get_dino().detect(
            image=image,
            text_query=target_phrase,
            box_threshold=self.args.dino_box_threshold,
            text_threshold=self.args.dino_text_threshold,
        )

        return select_top1_from_detections(
            detections=detections,
            method="DINO full phrase",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
        )

    def run_dino_category_top1(
        self,
        image_path: str,
        target_phrase: str,
        category: str,
        sample_id: Optional[str] = None,
        gt_box: Optional[List[float]] = None,
    ) -> SelectionResult:
        image = load_image(image_path)

        detections = self.get_dino().detect(
            image=image,
            text_query=category,
            box_threshold=self.args.dino_box_threshold,
            text_threshold=self.args.dino_text_threshold,
        )

        return select_top1_from_detections(
            detections=detections,
            method="DINO category top-1",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
        )

    def run_dino_clip_ranking(
        self,
        image_path: str,
        target_phrase: str,
        category: str,
        sample_id: Optional[str] = None,
        gt_box: Optional[List[float]] = None,
    ) -> SelectionResult:
        image = load_image(image_path)

        detections = self.get_dino().detect(
            image=image,
            text_query=category,
            box_threshold=self.args.dino_clip_candidate_box_threshold,
            text_threshold=self.args.dino_clip_candidate_text_threshold,
        )

        if len(detections) == 0:
            return SelectionResult(
                method="DINO + CLIP ranking",
                success=False,
                box=None,
                score=None,
                label=None,
                image=image_path,
                sample_id=sample_id,
                target_phrase=target_phrase,
                category=category,
                extra={"message": "No DINO category candidates."},
            )

        boxes = [d["box"] for d in detections]

        ranked = self.get_clip().rank(
            image=image,
            boxes=boxes,
            text_query=target_phrase,
            padding_ratio=self.args.clip_crop_padding,
        )

        if len(ranked) == 0:
            return SelectionResult(
                method="DINO + CLIP ranking",
                success=False,
                box=None,
                score=None,
                label=None,
                image=image_path,
                sample_id=sample_id,
                target_phrase=target_phrase,
                category=category,
                extra={"message": "CLIP ranking failed."},
            )

        reranked = []
        for rank_pos, (idx, clip_score) in enumerate(ranked):
            d = dict(detections[idx])
            d["dino_score"] = detections[idx].get("score")
            d["clip_score"] = clip_score
            d["score"] = clip_score
            d["rank"] = rank_pos
            reranked.append(d)

        return select_top1_from_detections(
            detections=reranked,
            method="DINO + CLIP ranking",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
            extra={
                "candidate_detector": "DINO category",
                "ranker": "CLIP image-text similarity",
            },
        )

    def run_owlv2(
        self,
        image_path: str,
        target_phrase: str,
        category: str,
        sample_id: Optional[str] = None,
        gt_box: Optional[List[float]] = None,
    ) -> SelectionResult:
        image = load_image(image_path)
        query = target_phrase if self.args.owl_query_mode == "phrase" else category

        detections = self.get_owlv2().detect(
            image=image,
            text_query=query,
            threshold=self.args.owl_threshold,
            add_photo_prefix=self.args.owl_add_photo_prefix,
        )

        return select_top1_from_detections(
            detections=detections,
            method="OWLv2",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
            extra={"owl_query": query},
        )

    def run_owlvit(
        self,
        image_path: str,
        target_phrase: str,
        category: str,
        sample_id: Optional[str] = None,
        gt_box: Optional[List[float]] = None,
    ) -> SelectionResult:
        image = load_image(image_path)
        query = target_phrase if self.args.owl_query_mode == "phrase" else category

        detections = self.get_owlvit().detect(
            image=image,
            text_query=query,
            threshold=self.args.owl_threshold,
            add_photo_prefix=self.args.owl_add_photo_prefix,
        )

        return select_top1_from_detections(
            detections=detections,
            method="OWL-ViT",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
            extra={"owl_query": query},
        )

    def run_kosmos2(
        self,
        image_path: str,
        target_phrase: str,
        category: str,
        sample_id: Optional[str] = None,
        gt_box: Optional[List[float]] = None,
    ) -> SelectionResult:
        image = load_image(image_path)

        detections = self.get_kosmos2().ground(
            image=image,
            target_phrase=target_phrase,
            category=category,
            max_new_tokens=self.args.kosmos2_max_new_tokens,
        )

        return select_top1_from_detections(
            detections=detections,
            method="KOSMOS-2 grounding",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
        )

    def run_method(
        self,
        method: str,
        image_path: str,
        target_phrase: str,
        category: str,
        sample_id: Optional[str] = None,
        gt_box: Optional[List[float]] = None,
    ) -> SelectionResult:
        if method == "spatial":
            return self.run_spatial_rule_only(image_path, target_phrase, category, sample_id, gt_box)
        if method == "dino_full_phrase":
            return self.run_dino_full_phrase(image_path, target_phrase, category, sample_id, gt_box)
        if method == "dino_category_top1":
            return self.run_dino_category_top1(image_path, target_phrase, category, sample_id, gt_box)
        if method == "dino_clip_ranking":
            return self.run_dino_clip_ranking(image_path, target_phrase, category, sample_id, gt_box)
        if method == "owlv2":
            return self.run_owlv2(image_path, target_phrase, category, sample_id, gt_box)
        if method == "owlvit":
            return self.run_owlvit(image_path, target_phrase, category, sample_id, gt_box)
        if method == "kosmos2":
            return self.run_kosmos2(image_path, target_phrase, category, sample_id, gt_box)

        raise ValueError(f"Unknown method: {method}")


# -------------------------
# Batch / CLI
# -------------------------

ALL_METHODS = [
    "spatial",
    "dino_full_phrase",
    "dino_category_top1",
    "dino_clip_ranking",
    "owlv2",
    "owlvit",
    "kosmos2",
]


def parse_methods(methods: List[str]) -> List[str]:
    if len(methods) == 1 and methods[0] == "all":
        return ALL_METHODS
    out = []
    for m in methods:
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method: {m}. Available: {ALL_METHODS} or all")
        out.append(m)
    return out


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_single(args):
    methods = parse_methods(args.methods)
    runner = BaselineRunner(args)

    gt_box = args.gt_box if args.gt_box else None

    results = []
    for method in methods:
        result = runner.run_method(
            method=method,
            image_path=args.image,
            target_phrase=args.target_phrase,
            category=args.category,
            sample_id=args.sample_id,
            gt_box=gt_box,
        )
        results.append(serialize_result(result))

        if args.visualize_dir:
            Path(args.visualize_dir).mkdir(parents=True, exist_ok=True)
            save_path = str(Path(args.visualize_dir) / f"{args.sample_id or 'single'}_{method}.jpg")
            draw_result(args.image, result, save_path)

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output:
        ensure_parent(args.output)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def run_batch(args):
    methods = parse_methods(args.methods)
    records = read_jsonl(args.input_jsonl)
    runner = BaselineRunner(args)

    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in records:
            image_path = rec["image"]
            target_phrase = rec["target_phrase"]
            category = rec["category"]
            sample_id = rec.get("id") or rec.get("sample_id")
            gt_box = rec.get("gt_box")

            for method in methods:
                result = runner.run_method(
                    method=method,
                    image_path=image_path,
                    target_phrase=target_phrase,
                    category=category,
                    sample_id=sample_id,
                    gt_box=gt_box,
                )
                row = serialize_result(result)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

                if args.visualize_dir:
                    Path(args.visualize_dir).mkdir(parents=True, exist_ok=True)
                    safe_id = str(sample_id or Path(image_path).stem)
                    save_path = str(Path(args.visualize_dir) / f"{safe_id}_{method}.jpg")
                    draw_result(image_path, result, save_path)


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Run target-selection baselines for multi-object image editing."
    )

    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
    )

    # single mode
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--target_phrase", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--gt_box", type=float, nargs=4, default=None)

    # batch mode
    parser.add_argument("--input_jsonl", type=str, default=None)

    # output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--visualize_dir", type=str, default=None)

    # methods
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help=f"Choose from {ALL_METHODS}, or use all.",
    )

    # device
    parser.add_argument("--device", type=str, default=None)

    # model ids
    parser.add_argument(
        "--dino_model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--owlv2_model",
        type=str,
        default="google/owlv2-base-patch16-ensemble",
    )
    parser.add_argument(
        "--owlvit_model",
        type=str,
        default="google/owlvit-base-patch32",
    )
    parser.add_argument(
        "--kosmos2_model",
        type=str,
        default="microsoft/kosmos-2-patch14-224",
    )

    # thresholds
    parser.add_argument("--dino_box_threshold", type=float, default=0.25)
    parser.add_argument("--dino_text_threshold", type=float, default=0.25)

    parser.add_argument("--dino_clip_candidate_box_threshold", type=float, default=0.20)
    parser.add_argument("--dino_clip_candidate_text_threshold", type=float, default=0.20)
    parser.add_argument("--clip_crop_padding", type=float, default=0.05)

    parser.add_argument("--owl_threshold", type=float, default=0.10)
    parser.add_argument(
        "--owl_query_mode",
        choices=["phrase", "category"],
        default="phrase",
        help="phrase: use target_phrase, category: use base category.",
    )
    parser.add_argument(
        "--owl_add_photo_prefix",
        action="store_true",
        help="Use query like 'a photo of {phrase}' for OWL models.",
    )

    parser.add_argument("--kosmos2_max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--kosmos2_dtype",
        type=str,
        default=None,
        choices=[None, "float16", "bfloat16", "float32"],
        help="Use float16/bfloat16 on GPU to reduce memory. Default keeps model default dtype.",
    )

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.mode == "single":
        if not args.image or not args.target_phrase or not args.category:
            raise ValueError(
                "single mode requires --image, --target_phrase, and --category"
            )
        run_single(args)
    else:
        if not args.input_jsonl:
            raise ValueError("batch mode requires --input_jsonl")
        if not args.output:
            raise ValueError("batch mode requires --output")
        run_batch(args)


if __name__ == "__main__":
    main()
