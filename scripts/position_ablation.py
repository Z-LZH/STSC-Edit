#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Position-oriented ablation for target selection.

Methods:
1. ours_wo_spatial_parsing
2. ours_wo_directional_ranking
3. ours_wo_anchor_reasoning
4. ours_full

这个脚本不修改 scripts/ablation.py，只复用其中已有的模型加载和基础工具。
"""

import argparse
import json
import os
import re
from pathlib import Path

import selection_baselines as base


# =========================
# Helper functions
# =========================

def center_prior_score(box, image_width, image_height):
    cx, cy = base.box_center(box)
    nx = cx / max(1.0, image_width)
    ny = cy / max(1.0, image_height)
    return float(1.0 - (abs(nx - 0.5) + abs(ny - 0.5)))


def parse_anchor_expression(text):
    """
    当前主要处理：
        the shirt worn by the left man

    返回：
        {
          "relation": "worn by",
          "anchor_spatial": "left",
          "anchor_category": "man"
        }
    """
    text_l = text.lower().strip()

    patterns = [
        r"\b(worn by|held by|carried by)\s+(?:the\s+)?(?:(left|right|top|bottom|upper|lower)\s+)?([a-z]+)",
        r"\b(on|near|next to|beside)\s+(?:the\s+)?(?:(left|right|top|bottom|upper|lower)\s+)?([a-z]+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text_l)
        if m:
            relation = m.group(1)
            spatial = m.group(2)
            anchor_category = m.group(3)

            if spatial == "upper":
                spatial = "top"
            if spatial == "lower":
                spatial = "bottom"

            return {
                "relation": relation,
                "anchor_spatial": spatial,
                "anchor_category": anchor_category,
            }

    return None


def rank_axis_aware(detections, image, target_phrase):
    """
    完整方向排序：
        left/right -> x 轴
        top/bottom -> y 轴
    """
    width, height = image.size
    spatial_terms = base.infer_spatial_terms(target_phrase)

    reranked = []
    for det in detections:
        d = dict(det)
        s = base.spatial_rule_score(
            d["box"],
            width,
            height,
            spatial_terms,
        )
        d["spatial_score"] = s
        d["score"] = s
        reranked.append(d)

    return sorted(reranked, key=lambda x: x["score"], reverse=True)


def rank_without_directional_axis(detections, image, target_phrase):
    """
    Ours w/o directional ranking:
    知道有 left/top 等位置词，但不把它们映射到 x/y 方向轴。
    这里退化成中心先验。
    """
    width, height = image.size
    spatial_terms = base.infer_spatial_terms(target_phrase)

    reranked = []
    for det in detections:
        d = dict(det)

        if len(spatial_terms) > 0:
            s = center_prior_score(d["box"], width, height)
        else:
            s = float(d.get("score", 0.0))

        d["center_prior_score"] = s
        d["score"] = s
        reranked.append(d)

    return sorted(reranked, key=lambda x: x["score"], reverse=True)


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def anchor_target_score(target_box, anchor_box, image_width, image_height):
    """
    target 越落在 anchor 框内、越接近 anchor 中心，分数越高。
    用于 shirt worn by the left man。
    """
    inter = intersection_area(target_box, anchor_box)
    target_area = max(1.0, base.box_area(target_box))
    containment = inter / target_area

    tx, ty = base.box_center(target_box)
    ax, ay = base.box_center(anchor_box)

    diag = max(1.0, (image_width ** 2 + image_height ** 2) ** 0.5)
    dist = ((tx - ax) ** 2 + (ty - ay) ** 2) ** 0.5 / diag
    proximity = 1.0 - dist

    return float(2.0 * containment + proximity)


# =========================
# Position Ablation Runner
# =========================

class PositionAblationRunner(base.BaselineRunner):
    def detect_category(self, image, category):
        return self.get_dino().detect(
            image=image,
            text_query=category,
            box_threshold=self.args.dino_box_threshold,
            text_threshold=self.args.dino_text_threshold,
        )

    def select_anchor(self, image, anchor_category, anchor_spatial, use_directional=True):
        anchor_dets = self.detect_category(image, anchor_category)

        if len(anchor_dets) == 0:
            return None

        if anchor_spatial is None:
            return sorted(anchor_dets, key=lambda x: x.get("score", 0.0), reverse=True)[0]

        pseudo_phrase = f"the {anchor_spatial} {anchor_category}"

        if use_directional:
            ranked = rank_axis_aware(anchor_dets, image, pseudo_phrase)
        else:
            ranked = rank_without_directional_axis(anchor_dets, image, pseudo_phrase)

        return ranked[0] if len(ranked) > 0 else None

    def run_ours_full(self, image_path, target_phrase, category, sample_id=None, gt_box=None):
        """
        Ours full:
        类别检测 + 空间词解析 + x/y 方向排序 + anchor 辅助定位。
        """
        image = base.load_image(image_path)
        width, height = image.size

        target_dets = self.detect_category(image, category)

        if len(target_dets) == 0:
            return base.SelectionResult(
                method="Ours full",
                success=False,
                box=None,
                score=None,
                label=None,
                image=image_path,
                sample_id=sample_id,
                target_phrase=target_phrase,
                category=category,
                extra={"message": "No target candidates."},
            )

        anchor_info = parse_anchor_expression(target_phrase)

        if anchor_info is not None:
            anchor_det = self.select_anchor(
                image=image,
                anchor_category=anchor_info["anchor_category"],
                anchor_spatial=anchor_info["anchor_spatial"],
                use_directional=True,
            )

            if anchor_det is not None:
                reranked = []
                for det in target_dets:
                    d = dict(det)
                    s = anchor_target_score(
                        target_box=d["box"],
                        anchor_box=anchor_det["box"],
                        image_width=width,
                        image_height=height,
                    )
                    d["anchor_target_score"] = s
                    d["score"] = s
                    reranked.append(d)

                reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

                return base.select_top1_from_detections(
                    detections=reranked,
                    method="Ours full",
                    image_path=image_path,
                    sample_id=sample_id,
                    target_phrase=target_phrase,
                    category=category,
                    gt_box=gt_box,
                    extra={
                        "selection_mode": "anchor_based",
                        "anchor_info": anchor_info,
                        "selected_anchor": anchor_det,
                    },
                )

        reranked = rank_axis_aware(target_dets, image, target_phrase)

        return base.select_top1_from_detections(
            detections=reranked,
            method="Ours full",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
            extra={
                "selection_mode": "axis_aware_spatial",
                "spatial_terms": base.infer_spatial_terms(target_phrase),
            },
        )

    def run_ours_wo_spatial_parsing(self, image_path, target_phrase, category, sample_id=None, gt_box=None):
        """
        Ours w/o spatial parsing:
        不使用 left/right/top/bottom 等空间词。
        只检测 category，并取 DINO 检测置信度最高的候选。
        """
        image = base.load_image(image_path)

        target_dets = self.detect_category(image, category)

        return base.select_top1_from_detections(
            detections=target_dets,
            method="Ours w/o spatial parsing",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
            extra={
                "removed_module": "spatial parsing",
                "note": "Spatial words are ignored; select highest-confidence category candidate.",
            },
        )

    def run_ours_wo_directional_ranking(self, image_path, target_phrase, category, sample_id=None, gt_box=None):
        """
        Ours w/o directional ranking:
        知道有 left/top 等位置词，但不按 x/y 方向轴排序。
        普通样本用中心先验；anchor 样本中 anchor 的选择也用中心先验。
        """
        image = base.load_image(image_path)
        width, height = image.size

        target_dets = self.detect_category(image, category)

        if len(target_dets) == 0:
            return base.SelectionResult(
                method="Ours w/o directional ranking",
                success=False,
                box=None,
                score=None,
                label=None,
                image=image_path,
                sample_id=sample_id,
                target_phrase=target_phrase,
                category=category,
                extra={"message": "No target candidates."},
            )

        anchor_info = parse_anchor_expression(target_phrase)

        if anchor_info is not None:
            anchor_det = self.select_anchor(
                image=image,
                anchor_category=anchor_info["anchor_category"],
                anchor_spatial=anchor_info["anchor_spatial"],
                use_directional=False,
            )

            if anchor_det is not None:
                reranked = []
                for det in target_dets:
                    d = dict(det)
                    s = anchor_target_score(
                        target_box=d["box"],
                        anchor_box=anchor_det["box"],
                        image_width=width,
                        image_height=height,
                    )
                    d["anchor_target_score"] = s
                    d["score"] = s
                    reranked.append(d)

                reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

                return base.select_top1_from_detections(
                    detections=reranked,
                    method="Ours w/o directional ranking",
                    image_path=image_path,
                    sample_id=sample_id,
                    target_phrase=target_phrase,
                    category=category,
                    gt_box=gt_box,
                    extra={
                        "removed_module": "directional ranking",
                        "anchor_info": anchor_info,
                        "selected_anchor": anchor_det,
                        "note": "Anchor spatial selection uses center prior instead of x/y axis.",
                    },
                )

        reranked = rank_without_directional_axis(target_dets, image, target_phrase)

        return base.select_top1_from_detections(
            detections=reranked,
            method="Ours w/o directional ranking",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
            extra={
                "removed_module": "directional ranking",
                "spatial_terms": base.infer_spatial_terms(target_phrase),
                "note": "Spatial words are detected but not mapped to x/y axis.",
            },
        )

    def run_ours_wo_anchor_reasoning(self, image_path, target_phrase, category, sample_id=None, gt_box=None):
        """
        Ours w/o anchor reasoning:
        对 shirt worn by the left man 这类表达，不先定位 left man。
        如果不是 anchor 表达，则和 full 一样使用正常空间排序。
        """
        image = base.load_image(image_path)

        target_dets = self.detect_category(image, category)

        if len(target_dets) == 0:
            return base.SelectionResult(
                method="Ours w/o anchor reasoning",
                success=False,
                box=None,
                score=None,
                label=None,
                image=image_path,
                sample_id=sample_id,
                target_phrase=target_phrase,
                category=category,
                extra={"message": "No target candidates."},
            )

        anchor_info = parse_anchor_expression(target_phrase)

        if anchor_info is not None:
            return base.select_top1_from_detections(
                detections=target_dets,
                method="Ours w/o anchor reasoning",
                image_path=image_path,
                sample_id=sample_id,
                target_phrase=target_phrase,
                category=category,
                gt_box=gt_box,
                extra={
                    "removed_module": "anchor reasoning",
                    "anchor_info": anchor_info,
                    "note": "Anchor is ignored; directly select target category candidate.",
                },
            )

        reranked = rank_axis_aware(target_dets, image, target_phrase)

        return base.select_top1_from_detections(
            detections=reranked,
            method="Ours w/o anchor reasoning",
            image_path=image_path,
            sample_id=sample_id,
            target_phrase=target_phrase,
            category=category,
            gt_box=gt_box,
            extra={
                "removed_module": "anchor reasoning",
                "note": "No anchor expression detected; use normal spatial ranking.",
            },
        )

    def run_method(self, method, image_path, target_phrase, category, sample_id=None, gt_box=None):
        if method == "ours_wo_spatial_parsing":
            return self.run_ours_wo_spatial_parsing(image_path, target_phrase, category, sample_id, gt_box)
        if method == "ours_wo_directional_ranking":
            return self.run_ours_wo_directional_ranking(image_path, target_phrase, category, sample_id, gt_box)
        if method == "ours_wo_anchor_reasoning":
            return self.run_ours_wo_anchor_reasoning(image_path, target_phrase, category, sample_id, gt_box)
        if method == "ours_full":
            return self.run_ours_full(image_path, target_phrase, category, sample_id, gt_box)

        raise ValueError(f"Unknown method: {method}")


METHODS = [
    "ours_wo_spatial_parsing",
    "ours_wo_directional_ranking",
    "ours_wo_anchor_reasoning",
    "ours_full",
]


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_methods(methods):
    if len(methods) == 1 and methods[0] == "all":
        return METHODS
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method: {m}. Available: {METHODS} or all")
    return methods


def run_batch(args):
    methods = parse_methods(args.methods)
    records = read_jsonl(args.input_jsonl)

    runner = PositionAblationRunner(args)

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

                row = base.serialize_result(result)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

                if args.visualize_dir:
                    Path(args.visualize_dir).mkdir(parents=True, exist_ok=True)
                    safe_id = str(sample_id or Path(image_path).stem)
                    save_path = str(Path(args.visualize_dir) / f"{safe_id}_{method}.jpg")
                    base.draw_result(image_path, result, save_path)


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--visualize_dir", type=str, default=None)

    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--dino_model", type=str, default="checkpoints/grounding-dino-base")
    parser.add_argument("--clip_model", type=str, default="checkpoints/clip-vit-base-patch32")
    parser.add_argument("--owlv2_model", type=str, default="checkpoints/owlv2-base-patch16-ensemble")
    parser.add_argument("--kosmos2_model", type=str, default="checkpoints/kosmos-2-patch14-224")

    parser.add_argument("--dino_box_threshold", type=float, default=0.25)
    parser.add_argument("--dino_text_threshold", type=float, default=0.25)

    # 下面这些是为了兼容 ablation.BaselineRunner 的 args，虽然本脚本不用 CLIP/OWLv2/Kosmos2。
    parser.add_argument("--dino_clip_candidate_box_threshold", type=float, default=0.20)
    parser.add_argument("--dino_clip_candidate_text_threshold", type=float, default=0.20)
    parser.add_argument("--clip_crop_padding", type=float, default=0.05)
    parser.add_argument("--owl_threshold", type=float, default=0.10)
    parser.add_argument("--owl_query_mode", choices=["phrase", "category"], default="phrase")
    parser.add_argument("--owl_add_photo_prefix", action="store_true")
    parser.add_argument("--kosmos2_max_new_tokens", type=int, default=64)
    parser.add_argument("--kosmos2_dtype", type=str, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
