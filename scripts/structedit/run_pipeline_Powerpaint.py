#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StructEdit full pipeline wrapper for PowerPaint.

This version keeps the original/v1 StructEdit step scripts unchanged and fixes
MOVE only in this wrapper.

MOVE execution:
1. Detect/select/segment with the original v1 pipeline.
2. Use PowerPaint object-removal to erase the source object.
3. Copy the original RGB pixels using the raw SAM mask.
4. Translate and alpha-composite those unchanged pixels at the destination.

Supported input modes:

1. PLE mode:
   python scripts/structedit/run_pipeline_Powerpaint.py \
     --input-json data/PLE_bench/0_random_140/export/annotations.json \
     --idx 1

2. Command mode:
   python scripts/structedit/run_pipeline_Powerpaint.py \
     --image-id 63 \
     --cmd "move the knife left of the fork"
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageFilter = None
    PIL_AVAILABLE = False


if __name__ == "__main__":
    sys.path.insert(
        0,
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ),
    )


try:
    from structedit.config import ensure_dir
except Exception:
    def ensure_dir(path: str):
        if path:
            os.makedirs(path, exist_ok=True)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


# ============================================================
# Basic utilities
# ============================================================
def clean_phrase(x: Any) -> str:
    x = str(x).strip().lower()
    x = x.replace("’", "'")
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"^[\s,.;:!?]+|[\s,.;:!?]+$", "", x)
    x = re.sub(r"^(the|a|an)\s+", "", x)
    return x.strip()


def save_json(obj: Any, path: str):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotations(path: str) -> List[Dict[str, Any]]:
    data = load_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["annotations", "data", "samples"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported annotation json format: {path}")


def safe_filename(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    return x.strip("_") or "item"


def script_path(filename: str) -> str:
    return os.path.join(SCRIPT_DIR, filename)


def file_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def all_exist(paths: List[str]) -> bool:
    return all(file_exists(p) for p in paths)


def get_ple_output_root(input_json: str) -> str:
    """
    Convert:
      data/PLE_bench/0_random_140/export/annotations.json

    To:
      outputs/PLE_bench/0_random_140
    """
    parts = os.path.normpath(input_json).split(os.sep)

    if "PLE_bench" in parts:
        i = parts.index("PLE_bench")
        if i + 1 < len(parts):
            return os.path.join(
                "outputs",
                "PLE_bench",
                parts[i + 1],
            )

    return os.path.join("outputs", "PLE_bench")


def resolve_command_image_path(
    image_root: str,
    image_id: str,
    image_ext: str,
) -> str:
    image_id = str(image_id)
    image_ext = image_ext.lstrip(".")
    return os.path.join(
        image_root,
        f"{image_id}.{image_ext}",
    )


# ============================================================
# Input context
# ============================================================
def infer_sample_context(args) -> Dict[str, Any]:
    has_ple = bool(args.input_json)
    has_command = bool(args.image_id or args.cmd)

    mode_count = int(has_ple) + int(has_command)

    if mode_count != 1:
        raise ValueError(
            "Use exactly one input mode:\n"
            "  PLE mode:     --input-json annotations.json --idx 0\n"
            "  Command mode: --image-id 63 --cmd "
            "\"move the knife left of the fork\""
        )

    # --------------------------------------------------------
    # Command mode
    # --------------------------------------------------------
    if has_command:
        if not args.image_id:
            raise ValueError(
                "Command mode requires --image-id."
            )

        if not args.cmd:
            raise ValueError(
                "Command mode requires --cmd."
            )

        sample_id = str(args.image_id)

        image_path = resolve_command_image_path(
            image_root=args.image_root,
            image_id=sample_id,
            image_ext=args.image_ext,
        )

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Command image not found: {image_path}"
            )

        sample_dir = os.path.join(
            "outputs",
            "command",
            sample_id,
        )
        parsed_json = os.path.join(
            sample_dir,
            "00_parsed_task.json",
        )

        return {
            "mode": "command",
            "sample_id": sample_id,
            "sample_dir": sample_dir,
            "parsed_json": parsed_json,
            "image_path": image_path,
            "cmd": args.cmd,
            "rule_parser_args": [
                "--image",
                image_path,
                "--cmd",
                args.cmd,
                "--sample-id",
                sample_id,
            ],
            "step_args": [
                "--parsed-json",
                parsed_json,
            ],
        }

    # --------------------------------------------------------
    # PLE mode
    # --------------------------------------------------------
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(
            f"Input JSON not found: {args.input_json}"
        )

    records = load_annotations(args.input_json)

    if args.idx < 0 or args.idx >= len(records):
        raise IndexError(
            f"--idx {args.idx} out of range, "
            f"total records={len(records)}"
        )

    record = records[args.idx]
    sample_id = str(record.get("id", "unknown"))

    out_root = get_ple_output_root(args.input_json)
    sample_dir = os.path.join(out_root, sample_id)
    parsed_json = os.path.join(
        sample_dir,
        "00_parsed_task.json",
    )

    return {
        "mode": "ple",
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "parsed_json": parsed_json,
        "image_path": record.get("image", ""),
        "cmd": "",
        "rule_parser_args": [
            "--input-json",
            args.input_json,
            "--idx",
            str(args.idx),
        ],
        "step_args": [
            "--input-json",
            args.input_json,
            "--idx",
            str(args.idx),
        ],
    }


# ============================================================
# Run existing step scripts
# ============================================================
def run_command(cmd: List[str], cwd: str):
    print("\n" + "=" * 100)
    print("[RUN]", " ".join(cmd))
    print("=" * 100)

    subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
    )


def run_step_if_needed(
    step_name: str,
    script_filename: str,
    step_args: List[str],
    expected_outputs: List[str],
    force: bool = False,
    skip: bool = False,
):
    if skip:
        print(f"[{step_name}] skipped.")
        return

    if (
        expected_outputs
        and all_exist(expected_outputs)
        and not force
    ):
        print(
            f"[{step_name}] existing outputs found, skip:"
        )
        for path in expected_outputs:
            print(f"  - {path}")
        return

    path = script_path(script_filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Step script not found: {path}"
        )

    cmd = [sys.executable, path] + step_args
    run_command(cmd, cwd=PROJECT_ROOT)


def run_structedit_steps(
    ctx: Dict[str, Any],
    args,
):
    sample_dir = ctx["sample_dir"]
    parsed_json = ctx["parsed_json"]
    step_args = ctx["step_args"]

    ensure_dir(sample_dir)

    # --------------------------------------------------------
    # Step 0: rule parser
    # --------------------------------------------------------
    run_step_if_needed(
        step_name="Step 0 Rule Parser",
        script_filename="rule_parser.py",
        step_args=ctx["rule_parser_args"],
        expected_outputs=[parsed_json],
        force=args.force_steps,
        skip=False,
    )

    if not os.path.exists(parsed_json):
        raise FileNotFoundError(
            f"Parsed json not found after rule parser: "
            f"{parsed_json}"
        )

    if args.skip_selection_steps:
        print(
            "[Selection Steps] skipped after rule parser "
            "because --skip-selection-steps is enabled."
        )
        return

    # --------------------------------------------------------
    # Step 1: DINO
    # --------------------------------------------------------
    dino_args = step_args + [
        "--box-threshold",
        str(args.box_threshold),
        "--text-threshold",
        str(args.text_threshold),
        "--max-per-object",
        str(args.max_per_object),
    ]

    if args.device:
        dino_args += ["--device", args.device]

    run_step_if_needed(
        step_name="Step 1 DINO",
        script_filename="dino_detector.py",
        step_args=dino_args,
        expected_outputs=[
            os.path.join(
                sample_dir,
                "01_dino_candidates.json",
            ),
        ],
        force=args.force_steps,
        skip=False,
    )

    # --------------------------------------------------------
    # Step 2: relation graph
    # --------------------------------------------------------
    run_step_if_needed(
        step_name="Step 2 Relation Graph",
        script_filename="relation_graph.py",
        step_args=step_args,
        expected_outputs=[
            os.path.join(
                sample_dir,
                "02_relation_nodes.json",
            ),
            os.path.join(
                sample_dir,
                "02_relation_edges.json",
            ),
        ],
        force=args.force_steps,
        skip=False,
    )

    # --------------------------------------------------------
    # Step 3: relation reasoner
    # --------------------------------------------------------
    run_step_if_needed(
        step_name="Step 3 Relation Reasoner",
        script_filename="relation_reasoner.py",
        step_args=step_args,
        expected_outputs=[
            os.path.join(
                sample_dir,
                "03_relation_decisions.json",
            ),
        ],
        force=args.force_steps,
        skip=False,
    )

    # --------------------------------------------------------
    # Step 4: context consistency
    # --------------------------------------------------------
    context_args = step_args + [
        "--context-weight",
        str(args.context_weight),
    ]

    run_step_if_needed(
        step_name="Step 4 Context Consistency",
        script_filename="context_consistency.py",
        step_args=context_args,
        expected_outputs=[
            os.path.join(
                sample_dir,
                "04_context_results.json",
            ),
            os.path.join(
                sample_dir,
                "04_context_decisions.json",
            ),
        ],
        force=args.force_steps,
        skip=False,
    )

    # --------------------------------------------------------
    # Step 5: target validator
    # --------------------------------------------------------
    validator_args = step_args + [
        "--accept-threshold",
        str(args.accept_threshold),
        "--min-det-score",
        str(args.min_det_score),
        "--min-relation-score",
        str(args.min_relation_score),
        "--min-context-score",
        str(args.min_context_score),
        "--min-area-ratio",
        str(args.min_area_ratio),
        "--max-area-ratio",
        str(args.max_area_ratio),
    ]

    if args.enable_clip:
        validator_args.append("--enable-clip")

    if args.clip_path:
        validator_args += [
            "--clip-path",
            args.clip_path,
        ]

    if args.device:
        validator_args += [
            "--device",
            args.device,
        ]

    run_step_if_needed(
        step_name="Step 5 Target Validator",
        script_filename="target_validator.py",
        step_args=validator_args,
        expected_outputs=[
            os.path.join(
                sample_dir,
                "05_verification_results.json",
            ),
        ],
        force=args.force_steps,
        skip=False,
    )

    # --------------------------------------------------------
    # Step 6: SAM2 masker
    # --------------------------------------------------------
    parsed_for_sam = load_json(parsed_json)
    primary_op_for_sam = infer_primary_edit_operation(
        parsed_for_sam
    )

    # MOVE copy needs a raw, non-dilated object alpha mask.
    # Deletion-mask expansion is applied later before PowerPaint.
    sam_mask_expand_pixels = (
        0
        if primary_op_for_sam == "move"
        else args.mask_expand_pixels
    )

    sam_args = step_args + [
        "--mask-expand-pixels",
        str(sam_mask_expand_pixels),
        "--box-expand-pixels",
        str(args.box_expand_pixels),
    ]

    if args.device:
        sam_args += [
            "--device",
            args.device,
        ]

    run_step_if_needed(
        step_name="Step 6 SAM2 Masker",
        script_filename=args.sam_script,
        step_args=sam_args,
        expected_outputs=[
            os.path.join(
                sample_dir,
                "06_sam2_masks",
                "06_sam2_mask_meta.json",
            ),
        ],
        force=args.force_steps,
        skip=False,
    )


# ============================================================
# Task / mask helpers
# ============================================================
def infer_primary_edit_operation(
    parsed_task: Dict[str, Any],
) -> str:
    """
    Return:
      delete / replace / move / add / color / material / ""
    """
    for unit in parsed_task.get(
        "edit_units",
        [],
    ) or []:
        op = clean_phrase(
            unit.get("operation", "")
        )
        edit_type = int(
            unit.get("edit_type", -1)
        )

        if (
            op in ["delete", "remove"]
            or edit_type == 2
        ):
            return "delete"

        if (
            op == "replace"
            or edit_type == 1
        ):
            return "replace"

        if (
            op in ["move", "relation"]
            or edit_type == 3
        ):
            return "move"

        if (
            op == "add"
            or edit_type == 0
        ):
            return "add"

        if (
            op == "color"
            or edit_type == 4
        ):
            return "color"

        if (
            op == "material"
            or edit_type == 5
        ):
            return "material"

    return ""


def is_delete_task(
    parsed_task: Dict[str, Any],
) -> bool:
    for unit in parsed_task.get(
        "edit_units",
        [],
    ) or []:
        op = clean_phrase(
            unit.get("operation", "")
        )
        edit_type = int(
            unit.get("edit_type", -1)
        )

        if (
            op in ["delete", "remove"]
            or edit_type == 2
        ):
            return True

    return False


def build_powerpaint_prompt(
    parsed_task: Dict[str, Any],
) -> str:
    if is_delete_task(parsed_task):
        return ""

    target_prompt = parsed_task.get(
        "target_prompt",
        "",
    )

    if target_prompt:
        return target_prompt

    return (
        "A realistic image with the requested edit applied. "
        "Preserve the original scene, lighting, perspective, "
        "and image quality."
    )


def infer_powerpaint_task_type(
    parsed_task: Dict[str, Any],
    default_task_type: str,
) -> str:
    if is_delete_task(parsed_task):
        return "object-removal"

    return default_task_type


def resolve_mask_path(
    sample_dir: str,
) -> str:
    mask_dir = os.path.join(
        sample_dir,
        "06_sam2_masks",
    )
    combined = os.path.join(
        mask_dir,
        "combined_mask.png",
    )

    if os.path.exists(combined):
        return combined

    if os.path.isdir(mask_dir):
        mask_candidates = [
            os.path.join(mask_dir, name)
            for name in os.listdir(mask_dir)
            if name.endswith("_mask.png")
        ]

        mask_candidates = [
            path
            for path in mask_candidates
            if os.path.exists(path)
        ]

        if len(mask_candidates) == 1:
            return mask_candidates[0]

    raise FileNotFoundError(
        "Cannot find SAM mask. "
        f"Expected combined mask: {combined}"
    )


def expand_mask_if_needed(
    mask_path: str,
    out_path: str,
    expand_px: int,
) -> str:
    if expand_px <= 0:
        shutil.copyfile(
            mask_path,
            out_path,
        )
        return out_path

    if not PIL_AVAILABLE:
        raise ImportError(
            "Pillow/PIL is required to expand edit mask."
        )

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.point(
        lambda value: 255
        if value > 127
        else 0
    )

    kernel_size = (
        2 * int(expand_px) + 1
    )
    mask = mask.filter(
        ImageFilter.MaxFilter(kernel_size)
    )

    ensure_dir(
        os.path.dirname(out_path)
    )
    mask.save(out_path)

    print(
        f"[Edit Mask] expanded by {expand_px}px"
    )
    print(
        f"[Edit Mask] saved: {out_path}"
    )

    return out_path


# ============================================================
# MOVE helpers
# ============================================================
def resolve_existing_output_path(
    path: str,
) -> str:
    if not path:
        return ""

    if os.path.exists(path):
        return os.path.abspath(path)

    project_relative = os.path.join(
        PROJECT_ROOT,
        path,
    )

    if os.path.exists(project_relative):
        return os.path.abspath(
            project_relative
        )

    return os.path.abspath(path)


def box_center(
    box: List[float],
) -> Tuple[float, float]:
    x1, y1, x2, y2 = [
        float(value)
        for value in box
    ]
    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
    )


def get_primary_move_unit(
    parsed_task: Dict[str, Any],
) -> Dict[str, Any]:
    for unit in parsed_task.get(
        "edit_units",
        [],
    ) or []:
        op = clean_phrase(
            unit.get("operation", "")
        )
        edit_type = int(
            unit.get("edit_type", -1)
        )

        if (
            op in ["move", "relation"]
            or edit_type == 3
        ):
            return unit

    raise RuntimeError(
        "MOVE operation was inferred, "
        "but no MOVE edit unit was found."
    )


def load_move_decision(
    sample_dir: str,
) -> Tuple[str, Dict[str, Any]]:
    path = resolve_existing_output_path(
        os.path.join(
            sample_dir,
            "03_relation_decisions.json",
        )
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MOVE decisions not found: {path}"
        )

    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError(
            f"Unexpected MOVE decisions format: {path}"
        )

    for key, decision in data.items():
        if not isinstance(decision, dict):
            continue

        op = clean_phrase(
            decision.get("operation", "")
        )

        if op in ["move", "relation"]:
            return key, decision

    # Compatibility fallback.
    for key, decision in data.items():
        if not isinstance(decision, dict):
            continue

        bbox = decision.get("bbox_xyxy")

        if (
            isinstance(bbox, list)
            and len(bbox) == 4
        ):
            return key, decision

    raise RuntimeError(
        f"No usable MOVE decision found in: {path}"
    )


def load_sam_meta(
    sample_dir: str,
) -> Dict[str, Any]:
    path = resolve_existing_output_path(
        os.path.join(
            sample_dir,
            "06_sam2_masks",
            "06_sam2_mask_meta.json",
        )
    )

    if not os.path.exists(path):
        return {}

    data = load_json(path)
    return data if isinstance(data, dict) else {}


def find_move_source_mask(
    sample_dir: str,
    decision_key: str,
    move_decision: Dict[str, Any],
    fallback_mask_path: str,
) -> str:
    meta = load_sam_meta(sample_dir)
    selected_uid = str(
        move_decision.get(
            "selected_uid",
            "",
        ) or ""
    )

    candidate_items = []

    direct = meta.get(decision_key)
    if isinstance(direct, dict):
        candidate_items.append(direct)

    for item in meta.values():
        if not isinstance(item, dict):
            continue

        item_selected_uid = str(
            item.get(
                "selected_uid",
                "",
            ) or ""
        )

        nested_decision = item.get(
            "decision",
            {},
        )

        nested_selected_uid = ""
        if isinstance(nested_decision, dict):
            nested_selected_uid = str(
                nested_decision.get(
                    "selected_uid",
                    "",
                ) or ""
            )

        if (
            selected_uid
            and selected_uid
            in [
                item_selected_uid,
                nested_selected_uid,
            ]
        ):
            candidate_items.append(item)

    for item in candidate_items:
        for key in [
            "source_raw_mask_path",
            "source_mask_path",
            "mask_path",
        ]:
            path = resolve_existing_output_path(
                str(item.get(key, "") or "")
            )

            if path and os.path.exists(path):
                return path

    fallback = resolve_existing_output_path(
        fallback_mask_path
    )

    if fallback and os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        "Cannot find the individual SAM source mask "
        f"for MOVE decision {decision_key}."
    )


def normalize_dino_candidates(
    data: Any,
) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    if isinstance(data, dict):
        for key in [
            "candidates",
            "detections",
            "items",
        ]:
            value = data.get(key)
            if isinstance(value, list):
                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

        values = [
            value
            for value in data.values()
            if isinstance(value, dict)
        ]

        if values:
            return values

    return []


def load_dino_candidates(
    sample_dir: str,
) -> List[Dict[str, Any]]:
    path = resolve_existing_output_path(
        os.path.join(
            sample_dir,
            "01_dino_candidates.json",
        )
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DINO candidates not found: {path}"
        )

    candidates = normalize_dino_candidates(
        load_json(path)
    )

    if not candidates:
        raise RuntimeError(
            f"No DINO candidates found in: {path}"
        )

    return candidates


def candidate_object_name(
    candidate: Dict[str, Any],
) -> str:
    for key in [
        "object_name",
        "query_object",
        "object_text",
        "phrase",
    ]:
        value = clean_phrase(
            candidate.get(key, "")
        )

        if value:
            return value

    return ""


def select_move_anchor_candidate(
    move_unit: Dict[str, Any],
    move_decision: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    anchor_object = clean_phrase(
        move_unit.get(
            "anchor_object",
            "",
        )
    )

    if not anchor_object:
        raise RuntimeError(
            "MOVE edit unit has no anchor_object."
        )

    matched_uid = str(
        move_decision.get(
            "matched_partner_uid",
            "",
        ) or ""
    )

    if matched_uid:
        for candidate in candidates:
            if (
                str(candidate.get("uid", ""))
                == matched_uid
                and candidate_object_name(candidate)
                == anchor_object
            ):
                return candidate

    anchor_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate_object_name(candidate)
            == anchor_object
            and isinstance(
                candidate.get("bbox_xyxy"),
                list,
            )
            and len(
                candidate.get("bbox_xyxy")
            ) == 4
        )
    ]

    if not anchor_candidates:
        raise RuntimeError(
            "No DINO candidate found "
            f"for MOVE anchor: {anchor_object}"
        )

    source_bbox = move_decision.get(
        "bbox_xyxy"
    )

    if (
        not isinstance(source_bbox, list)
        or len(source_bbox) != 4
    ):
        return max(
            anchor_candidates,
            key=lambda candidate: float(
                candidate.get(
                    "score",
                    candidate.get(
                        "base_det_score",
                        0.0,
                    ),
                )
            ),
        )

    source_cx, source_cy = box_center(
        source_bbox
    )

    def rank(
        candidate: Dict[str, Any],
    ) -> float:
        cx, cy = box_center(
            candidate["bbox_xyxy"]
        )
        distance_sq = (
            (cx - source_cx) ** 2
            + (cy - source_cy) ** 2
        )
        score = float(
            candidate.get(
                "score",
                candidate.get(
                    "base_det_score",
                    0.0,
                ),
            )
        )

        return (
            distance_sq
            - 2500.0 * score
        )

    return min(
        anchor_candidates,
        key=rank,
    )


def compute_move_destination_box(
    source_bbox: List[float],
    anchor_bbox: List[float],
    relation: str,
    image_width: int,
    image_height: int,
) -> List[float]:
    sx1, sy1, sx2, sy2 = [
        float(value)
        for value in source_bbox
    ]
    ax1, ay1, ax2, ay2 = [
        float(value)
        for value in anchor_bbox
    ]

    object_width = max(
        1.0,
        sx2 - sx1,
    )
    object_height = max(
        1.0,
        sy2 - sy1,
    )

    anchor_width = max(
        1.0,
        ax2 - ax1,
    )
    anchor_height = max(
        1.0,
        ay2 - ay1,
    )

    anchor_cx = 0.5 * (
        ax1 + ax2
    )
    anchor_cy = 0.5 * (
        ay1 + ay2
    )

    gap = max(
        6.0,
        0.12 * max(
            anchor_width,
            anchor_height,
        ),
    )

    relation = clean_phrase(relation)

    if relation == "left_of":
        x1 = (
            ax1
            - gap
            - object_width
        )
        y1 = (
            anchor_cy
            - object_height / 2.0
        )

    elif relation == "right_of":
        x1 = ax2 + gap
        y1 = (
            anchor_cy
            - object_height / 2.0
        )

    elif relation == "above":
        x1 = (
            anchor_cx
            - object_width / 2.0
        )
        y1 = (
            ay1
            - gap
            - object_height
        )

    elif relation == "below":
        x1 = (
            anchor_cx
            - object_width / 2.0
        )
        y1 = ay2 + gap

    elif relation in [
        "near",
        "beside",
    ]:
        if (
            image_width - ax2
            >= ax1
        ):
            x1 = ax2 + gap
        else:
            x1 = (
                ax1
                - gap
                - object_width
            )

        y1 = (
            anchor_cy
            - object_height / 2.0
        )

    elif relation in [
        "inside",
        "in",
        "on",
    ]:
        x1 = (
            anchor_cx
            - object_width / 2.0
        )
        y1 = (
            anchor_cy
            - object_height / 2.0
        )

    else:
        x1 = (
            anchor_cx
            - object_width / 2.0
            + 0.15 * anchor_width
        )
        y1 = (
            anchor_cy
            - object_height / 2.0
            + 0.10 * anchor_height
        )

    max_x1 = max(
        0.0,
        float(image_width) - object_width,
    )
    max_y1 = max(
        0.0,
        float(image_height) - object_height,
    )

    x1 = max(
        0.0,
        min(max_x1, x1),
    )
    y1 = max(
        0.0,
        min(max_y1, y1),
    )

    return [
        float(x1),
        float(y1),
        float(x1 + object_width),
        float(y1 + object_height),
    ]


def paste_rgba_with_clipping(
    canvas: Image.Image,
    crop: Image.Image,
    destination_xy: Tuple[int, int],
):
    dest_x, dest_y = destination_xy
    canvas_width, canvas_height = canvas.size
    crop_width, crop_height = crop.size

    dest_left = max(
        0,
        dest_x,
    )
    dest_top = max(
        0,
        dest_y,
    )
    dest_right = min(
        canvas_width,
        dest_x + crop_width,
    )
    dest_bottom = min(
        canvas_height,
        dest_y + crop_height,
    )

    if (
        dest_right <= dest_left
        or dest_bottom <= dest_top
    ):
        return

    src_left = dest_left - dest_x
    src_top = dest_top - dest_y
    src_right = (
        src_left
        + dest_right
        - dest_left
    )
    src_bottom = (
        src_top
        + dest_bottom
        - dest_top
    )

    clipped = crop.crop(
        (
            src_left,
            src_top,
            src_right,
            src_bottom,
        )
    )

    canvas.alpha_composite(
        clipped,
        dest=(
            dest_left,
            dest_top,
        ),
    )


def paste_mask_with_clipping(
    canvas: Image.Image,
    crop: Image.Image,
    destination_xy: Tuple[int, int],
):
    dest_x, dest_y = destination_xy
    canvas_width, canvas_height = canvas.size
    crop_width, crop_height = crop.size

    dest_left = max(
        0,
        dest_x,
    )
    dest_top = max(
        0,
        dest_y,
    )
    dest_right = min(
        canvas_width,
        dest_x + crop_width,
    )
    dest_bottom = min(
        canvas_height,
        dest_y + crop_height,
    )

    if (
        dest_right <= dest_left
        or dest_bottom <= dest_top
    ):
        return

    src_left = dest_left - dest_x
    src_top = dest_top - dest_y
    src_right = (
        src_left
        + dest_right
        - dest_left
    )
    src_bottom = (
        src_top
        + dest_bottom
        - dest_top
    )

    clipped = crop.crop(
        (
            src_left,
            src_top,
            src_right,
            src_bottom,
        )
    )

    canvas.paste(
        clipped,
        (
            dest_left,
            dest_top,
        ),
    )


def create_move_paste_layer(
    image_path: str,
    source_mask_path: str,
    source_bbox: List[float],
    destination_bbox: List[float],
    output_dir: str,
) -> Dict[str, Any]:
    image = Image.open(
        image_path
    ).convert("RGB")

    source_mask = Image.open(
        source_mask_path
    ).convert("L")

    if source_mask.size != image.size:
        source_mask = source_mask.resize(
            image.size,
            Image.NEAREST,
        )

    source_mask = source_mask.point(
        lambda value: 255
        if value > 127
        else 0
    )

    source_cx, source_cy = box_center(
        source_bbox
    )
    destination_cx, destination_cy = box_center(
        destination_bbox
    )

    dx = int(
        round(
            destination_cx
            - source_cx
        )
    )
    dy = int(
        round(
            destination_cy
            - source_cy
        )
    )

    extent = source_mask.getbbox()

    if extent is None:
        raise RuntimeError(
            f"Empty MOVE source mask: {source_mask_path}"
        )

    object_crop = image.crop(
        extent
    ).convert("RGBA")

    alpha_crop = source_mask.crop(
        extent
    )

    object_crop.putalpha(
        alpha_crop
    )

    destination_xy = (
        int(extent[0] + dx),
        int(extent[1] + dy),
    )

    paste_layer = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    paste_rgba_with_clipping(
        paste_layer,
        object_crop,
        destination_xy,
    )

    destination_mask = Image.new(
        "L",
        image.size,
        0,
    )

    paste_mask_with_clipping(
        destination_mask,
        alpha_crop,
        destination_xy,
    )

    paste_layer_path = os.path.join(
        output_dir,
        "move_paste_layer_rgba.png",
    )
    destination_mask_path = os.path.join(
        output_dir,
        "move_destination_mask.png",
    )

    paste_layer.save(
        paste_layer_path
    )
    destination_mask.save(
        destination_mask_path
    )

    return {
        "source_mask_path": source_mask_path,
        "source_bbox_xyxy": [
            float(value)
            for value in source_bbox
        ],
        "destination_bbox_xyxy": [
            float(value)
            for value in destination_bbox
        ],
        "translation_dx_dy": [
            dx,
            dy,
        ],
        "paste_layer_rgba_path": (
            paste_layer_path
        ),
        "destination_mask_path": (
            destination_mask_path
        ),
    }


def prepare_move_plan(
    parsed_task: Dict[str, Any],
    sample_dir: str,
    image_path: str,
    fallback_mask_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    move_unit = get_primary_move_unit(
        parsed_task
    )

    decision_key, move_decision = (
        load_move_decision(sample_dir)
    )

    source_bbox = move_decision.get(
        "bbox_xyxy"
    )

    if (
        not isinstance(source_bbox, list)
        or len(source_bbox) != 4
    ):
        raise RuntimeError(
            "MOVE source bbox missing in decision: "
            f"{decision_key}"
        )

    source_mask_path = find_move_source_mask(
        sample_dir=sample_dir,
        decision_key=decision_key,
        move_decision=move_decision,
        fallback_mask_path=fallback_mask_path,
    )

    candidates = load_dino_candidates(
        sample_dir
    )

    anchor_candidate = (
        select_move_anchor_candidate(
            move_unit=move_unit,
            move_decision=move_decision,
            candidates=candidates,
        )
    )

    anchor_bbox = anchor_candidate.get(
        "bbox_xyxy"
    )

    relation = clean_phrase(
        move_unit.get(
            "relation",
            "",
        )
    )

    with Image.open(
        image_path
    ) as image:
        image_width, image_height = (
            image.size
        )

    destination_bbox = (
        compute_move_destination_box(
            source_bbox=source_bbox,
            anchor_bbox=anchor_bbox,
            relation=relation,
            image_width=image_width,
            image_height=image_height,
        )
    )

    layer_info = create_move_paste_layer(
        image_path=image_path,
        source_mask_path=source_mask_path,
        source_bbox=source_bbox,
        destination_bbox=destination_bbox,
        output_dir=output_dir,
    )

    return {
        "decision_key": decision_key,
        "source_object": clean_phrase(
            move_decision.get(
                "source_object",
                move_unit.get(
                    "target_object",
                    "",
                ),
            )
        ),
        "anchor_object": clean_phrase(
            move_unit.get(
                "anchor_object",
                "",
            )
        ),
        "anchor_uid": anchor_candidate.get(
            "uid",
            "",
        ),
        "anchor_bbox_xyxy": [
            float(value)
            for value in anchor_bbox
        ],
        "relation": relation,
        **layer_info,
    }


def composite_move_paste_layer(
    inpainted_path: str,
    paste_layer_path: str,
    output_path: str,
):
    base = Image.open(
        inpainted_path
    ).convert("RGBA")

    paste_layer = Image.open(
        paste_layer_path
    ).convert("RGBA")

    if paste_layer.size != base.size:
        raise ValueError(
            "MOVE paste layer size mismatch: "
            f"base={base.size}, "
            f"layer={paste_layer.size}"
        )

    result = Image.alpha_composite(
        base,
        paste_layer,
    ).convert("RGB")

    ensure_dir(
        os.path.dirname(output_path)
    )
    result.save(output_path)


# ============================================================
# PowerPaint adapter
# ============================================================
class PowerPaintFillEditor:
    """
    Minimal PowerPaint v2 / v2-1 adapter.

    PowerPaint convention:
      white mask = repaint
      black mask = keep
    """

    def __init__(
        self,
        powerpaint_repo_dir: str,
        checkpoint_dir: str,
        version: str = "ppt-v2",
        weight_dtype: str = "float16",
        guidance_scale: float = 12.0,
        num_inference_steps: int = 45,
        seed: int = 42,
        fitting_degree: float = 1.0,
        local_files_only: bool = True,
        preserve_unmasked: bool = True,
        mask_blur_px: int = 3,
    ):
        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow/PIL is required by "
                "PowerPaintFillEditor."
            )

        self.powerpaint_repo_dir = os.path.abspath(
            powerpaint_repo_dir
        )
        self.checkpoint_dir = os.path.abspath(
            checkpoint_dir
        )
        self.version = version
        self.guidance_scale = float(
            guidance_scale
        )
        self.num_inference_steps = int(
            num_inference_steps
        )
        self.seed = int(seed)
        self.fitting_degree = float(
            fitting_degree
        )
        self.local_files_only = bool(
            local_files_only
        )
        self.preserve_unmasked = bool(
            preserve_unmasked
        )
        self.mask_blur_px = int(
            mask_blur_px
        )

        app_py = os.path.join(
            self.powerpaint_repo_dir,
            "app.py",
        )

        if not os.path.exists(app_py):
            raise FileNotFoundError(
                "PowerPaint app.py not found. "
                "Check --powerpaint-repo-dir: "
                f"{self.powerpaint_repo_dir}"
            )

        if not os.path.isdir(
            self.checkpoint_dir
        ):
            raise FileNotFoundError(
                "PowerPaint checkpoint dir not found. "
                "Check --powerpaint-checkpoint-dir: "
                f"{self.checkpoint_dir}"
            )

        if (
            self.powerpaint_repo_dir
            not in sys.path
        ):
            sys.path.insert(
                0,
                self.powerpaint_repo_dir,
            )

        import torch
        from app import PowerPaintController

        if weight_dtype == "float16":
            torch_dtype = torch.float16
        elif weight_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif weight_dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(
                "Unsupported weight dtype: "
                f"{weight_dtype}"
            )

        print(
            "[PowerPaint] Initializing "
            "PowerPaintController"
        )
        print(
            "[PowerPaint] repo_dir="
            f"{self.powerpaint_repo_dir}"
        )
        print(
            "[PowerPaint] checkpoint_dir="
            f"{self.checkpoint_dir}"
        )
        print(
            "[PowerPaint] version="
            f"{self.version}"
        )
        print(
            "[PowerPaint] local_files_only="
            f"{self.local_files_only}"
        )

        self.controller = PowerPaintController(
            weight_dtype=torch_dtype,
            checkpoint_dir=self.checkpoint_dir,
            local_files_only=self.local_files_only,
            version=self.version,
        )

        print(
            "[PowerPaint] Initialization finished."
        )

    def _load_image_and_mask(
        self,
        image_path: str,
        mask_path: str,
    ):
        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        if mask.size != image.size:
            mask = mask.resize(
                image.size,
                Image.NEAREST,
            )

        mask = mask.point(
            lambda value: 255
            if value > 127
            else 0
        )

        return image, mask

    def _composite_to_original_size(
        self,
        original: Image.Image,
        generated: Image.Image,
        mask: Image.Image,
    ):
        generated = generated.convert("RGB")

        if generated.size != original.size:
            generated = generated.resize(
                original.size,
                Image.BICUBIC,
            )

        if not self.preserve_unmasked:
            return generated

        alpha = mask.convert("L")

        if alpha.size != original.size:
            alpha = alpha.resize(
                original.size,
                Image.NEAREST,
            )

        if self.mask_blur_px > 0:
            alpha = alpha.filter(
                ImageFilter.GaussianBlur(
                    radius=self.mask_blur_px
                )
            )

        return Image.composite(
            generated,
            original.convert("RGB"),
            alpha,
        )

    def edit(
        self,
        image_path: str,
        mask_path: str,
        target_prompt: str,
        output_path: str,
        task_type: str,
        negative_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        image, mask = (
            self._load_image_and_mask(
                image_path,
                mask_path,
            )
        )

        if task_type in [
            "delete",
            "remove",
            "object-removal",
        ]:
            task = "object-removal"
            prompt = ""
            negative_prompt = (
                negative_prompt
                or (
                    "object, subject, foreground object, "
                    "person, animal, text, logo, watermark"
                )
            )
            removal_prompt = ""
            removal_negative_prompt = (
                negative_prompt
            )

        elif task_type == "shape-guided":
            task = "shape-guided"
            prompt = target_prompt or ""
            negative_prompt = (
                negative_prompt
                or (
                    "worst quality, low quality, blurry, "
                    "distorted, text, watermark, logo"
                )
            )
            removal_prompt = ""
            removal_negative_prompt = (
                negative_prompt
            )

        else:
            task = "text-guided"
            prompt = target_prompt or ""
            negative_prompt = (
                negative_prompt
                or (
                    "worst quality, low quality, blurry, "
                    "distorted, text, watermark, logo"
                )
            )
            removal_prompt = ""
            removal_negative_prompt = (
                negative_prompt
            )

        input_image = {
            "image": image,
            "mask": mask.convert("RGB"),
        }

        print("=" * 100)
        print("[PowerPaint Edit]")
        print(f"task={task}")
        print(f"prompt={prompt}")
        print(
            "negative_prompt="
            f"{negative_prompt}"
        )
        print(
            "guidance_scale="
            f"{self.guidance_scale}"
        )
        print(
            "steps="
            f"{self.num_inference_steps}"
        )
        print(f"seed={self.seed}")
        print("=" * 100)

        outputs, aux = self.controller.infer(
            input_image=input_image,
            text_guided_prompt=prompt,
            text_guided_negative_prompt=(
                negative_prompt
            ),
            shape_guided_prompt=prompt,
            shape_guided_negative_prompt=(
                negative_prompt
            ),
            fitting_degree=self.fitting_degree,
            ddim_steps=self.num_inference_steps,
            scale=self.guidance_scale,
            seed=self.seed,
            task=task,
            vertical_expansion_ratio=None,
            horizontal_expansion_ratio=None,
            outpaint_prompt="",
            outpaint_negative_prompt=(
                negative_prompt
            ),
            removal_prompt=removal_prompt,
            removal_negative_prompt=(
                removal_negative_prompt
            ),
        )

        if not outputs:
            raise RuntimeError(
                "PowerPaint returned no output image."
            )

        result = outputs[0].convert("RGB")

        result = (
            self._composite_to_original_size(
                image,
                result,
                mask,
            )
        )

        ensure_dir(
            os.path.dirname(output_path)
        )
        result.save(output_path)

        return {
            "backend": "powerpaint",
            "task": task,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_path": image_path,
            "mask_path": mask_path,
            "output_path": output_path,
            "checkpoint_dir": self.checkpoint_dir,
            "version": self.version,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": (
                self.num_inference_steps
            ),
            "seed": self.seed,
            "preserve_unmasked": (
                self.preserve_unmasked
            ),
            "mask_blur_px": self.mask_blur_px,
        }


# ============================================================
# Final PowerPaint edit
# ============================================================
def run_powerpaint_edit(
    ctx: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    sample_dir = ctx["sample_dir"]
    parsed_json = ctx["parsed_json"]
    mode = ctx["mode"]
    sample_id = ctx["sample_id"]

    if not os.path.exists(parsed_json):
        raise FileNotFoundError(
            f"Parsed json not found: {parsed_json}"
        )

    parsed_task = load_json(parsed_json)
    image_path = parsed_task.get(
        "image_path",
        "",
    )

    if (
        not image_path
        or not os.path.exists(image_path)
    ):
        raise FileNotFoundError(
            f"Image path not found: {image_path}"
        )

    primary_op = infer_primary_edit_operation(
        parsed_task
    )

    raw_mask_path = resolve_mask_path(
        sample_dir
    )

    output_dir = os.path.join(
        args.powerpaint_output_root,
        mode,
        str(sample_id),
    )
    ensure_dir(output_dir)

    move_plan = None

    if primary_op == "move":
        prompt = ""
        task_type = "object-removal"

        move_plan = prepare_move_plan(
            parsed_task=parsed_task,
            sample_dir=sample_dir,
            image_path=image_path,
            fallback_mask_path=raw_mask_path,
            output_dir=output_dir,
        )

        # PowerPaint receives only the original source-object mask.
        raw_mask_path = move_plan[
            "source_mask_path"
        ]

    else:
        prompt = build_powerpaint_prompt(
            parsed_task
        )

        task_type = (
            infer_powerpaint_task_type(
                parsed_task=parsed_task,
                default_task_type=(
                    args.powerpaint_task_type
                ),
            )
        )

    if primary_op == "delete":
        edit_expand_px = (
            args.delete_edit_mask_expand_pixels
        )
    elif primary_op == "replace":
        edit_expand_px = (
            args.replace_edit_mask_expand_pixels
        )
    elif primary_op == "move":
        edit_expand_px = (
            args.move_edit_mask_expand_pixels
        )
    elif task_type == "object-removal":
        edit_expand_px = (
            args.delete_edit_mask_expand_pixels
        )
    else:
        edit_expand_px = (
            args.edit_mask_expand_pixels
        )

    print(
        f"[Edit Mask] primary_op={primary_op}, "
        f"task_type={task_type}, "
        f"edit_expand_px={edit_expand_px}"
    )

    mask_used_path = os.path.join(
        output_dir,
        "mask_used.png",
    )

    edit_mask_path = expand_mask_if_needed(
        mask_path=raw_mask_path,
        out_path=mask_used_path,
        expand_px=edit_expand_px,
    )

    output_path = os.path.join(
        output_dir,
        "edited.png",
    )

    powerpaint_stage_output_path = (
        os.path.join(
            output_dir,
            "move_source_removed.png",
        )
        if primary_op == "move"
        else output_path
    )

    if (
        os.path.exists(output_path)
        and not args.force_edit
    ):
        print(
            "[PowerPaint] edited image exists, skip: "
            f"{output_path}"
        )

        result = {
            "skipped": True,
            "reason": (
                "edited image exists; "
                "use --force-edit to overwrite"
            ),
            "output_path": output_path,
            "mask_path": edit_mask_path,
        }

        save_json(
            result,
            os.path.join(
                output_dir,
                "edit_result.json",
            ),
        )
        return result

    editor = PowerPaintFillEditor(
        powerpaint_repo_dir=(
            args.powerpaint_repo_dir
        ),
        checkpoint_dir=(
            args.powerpaint_checkpoint_dir
        ),
        version=args.powerpaint_version,
        weight_dtype=(
            args.powerpaint_weight_dtype
        ),
        guidance_scale=(
            args.powerpaint_guidance
        ),
        num_inference_steps=(
            args.powerpaint_steps
        ),
        seed=args.powerpaint_seed,
        fitting_degree=(
            args.powerpaint_fitting_degree
        ),
        local_files_only=(
            not args.powerpaint_allow_download
        ),
        preserve_unmasked=(
            not args.powerpaint_no_preserve_unmasked
        ),
        mask_blur_px=(
            args.powerpaint_mask_blur_px
        ),
    )

    result = editor.edit(
        image_path=image_path,
        mask_path=edit_mask_path,
        target_prompt=prompt,
        output_path=(
            powerpaint_stage_output_path
        ),
        task_type=task_type,
        negative_prompt=(
            args.powerpaint_negative_prompt
            or None
        ),
    )

    if primary_op == "move":
        if move_plan is None:
            raise RuntimeError(
                "MOVE plan was not prepared."
            )

        composite_move_paste_layer(
            inpainted_path=(
                powerpaint_stage_output_path
            ),
            paste_layer_path=(
                move_plan[
                    "paste_layer_rgba_path"
                ]
            ),
            output_path=output_path,
        )

        result["output_path"] = (
            output_path
        )
        result[
            "powerpaint_stage_output_path"
        ] = powerpaint_stage_output_path
        result["move_strategy"] = (
            "remove_source_then_paste_original_pixels"
        )
        result["move_plan"] = move_plan

    # --------------------------------------------------------
    # Save useful copies
    # --------------------------------------------------------
    shutil.copyfile(
        parsed_json,
        os.path.join(
            output_dir,
            "00_parsed_task.json",
        ),
    )

    verification_path = os.path.join(
        sample_dir,
        "05_verification_results.json",
    )
    if os.path.exists(verification_path):
        shutil.copyfile(
            verification_path,
            os.path.join(
                output_dir,
                "05_verification_results.json",
            ),
        )

    sam_meta_path = os.path.join(
        sample_dir,
        "06_sam2_masks",
        "06_sam2_mask_meta.json",
    )
    if os.path.exists(sam_meta_path):
        shutil.copyfile(
            sam_meta_path,
            os.path.join(
                output_dir,
                "06_sam2_mask_meta.json",
            ),
        )

    raw_combined_mask = os.path.join(
        sample_dir,
        "06_sam2_masks",
        "combined_mask.png",
    )
    if os.path.exists(raw_combined_mask):
        shutil.copyfile(
            raw_combined_mask,
            os.path.join(
                output_dir,
                "combined_mask_raw.png",
            ),
        )

    save_json(
        result,
        os.path.join(
            output_dir,
            "edit_result.json",
        ),
    )

    summary = {
        "mode": mode,
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "powerpaint_output_dir": output_dir,
        "parsed_json": parsed_json,
        "image_path": image_path,
        "raw_mask_path": raw_mask_path,
        "mask_used_path": edit_mask_path,
        "edited_path": output_path,
        "task_type": task_type,
        "prompt": prompt,
        "primary_operation": primary_op,
        "move_plan": move_plan,
        "powerpaint_stage_output_path": (
            powerpaint_stage_output_path
        ),
        "powerpaint_result": result,
    }

    save_json(
        summary,
        os.path.join(
            output_dir,
            "pipeline_powerpaint_summary.json",
        ),
    )

    print("=" * 100)
    print("[PowerPaint Pipeline Finished]")
    print(
        f"edited image: {output_path}"
    )
    print(
        f"mask used:    {edit_mask_path}"
    )
    print(
        f"output dir:   {output_dir}"
    )
    print("=" * 100)

    return summary


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "StructEdit PowerPaint pipeline wrapper. "
            "It runs rule_parser, dino_detector, "
            "relation_graph, relation_reasoner, "
            "context_consistency, target_validator, "
            "sam_masker, then PowerPaint."
        )
    )

    # Input mode A: PLE
    parser.add_argument(
        "--input-json",
        type=str,
        default="",
        help="PLE annotation JSON path.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="PLE sample index.",
    )

    # Input mode B: command
    parser.add_argument(
        "--image-id",
        type=str,
        default="",
        help="Image id for command mode.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="data/vg/no_edit",
        help="Image root for command mode.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="jpg",
        help="Image extension for command mode.",
    )
    parser.add_argument(
        "--cmd",
        type=str,
        default="",
        help="English edit command.",
    )

    # Step control
    parser.add_argument(
        "--skip-selection-steps",
        action="store_true",
        help=(
            "Skip DINO/graph/reasoner/context/"
            "validator/SAM steps."
        ),
    )
    parser.add_argument(
        "--force-steps",
        action="store_true",
        help=(
            "Rerun all StructEdit intermediate steps "
            "even if outputs already exist."
        ),
    )
    parser.add_argument(
        "--force-edit",
        action="store_true",
        help="Overwrite existing edited.png.",
    )
    parser.add_argument(
        "--sam-script",
        type=str,
        default="sam_masker.py",
        help="SAM2 step script filename.",
    )

    # DINO
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--max-per-object",
        type=int,
        default=8,
    )

    # Context / validation
    parser.add_argument(
        "--context-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--min-det-score",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--min-relation-score",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-context-score",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0002,
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--enable-clip",
        action="store_true",
    )
    parser.add_argument(
        "--clip-path",
        type=str,
        default="",
    )

    # SAM2
    parser.add_argument(
        "--mask-expand-pixels",
        type=int,
        default=2,
        help=(
            "Pass to sam_masker.py. "
            "MOVE overrides this to 0 for copy alpha."
        ),
    )
    parser.add_argument(
        "--box-expand-pixels",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--edit-mask-expand-pixels",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--delete-edit-mask-expand-pixels",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--replace-edit-mask-expand-pixels",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--move-edit-mask-expand-pixels",
        type=int,
        default=8,
    )

    # PowerPaint
    parser.add_argument(
        "--powerpaint-output-root",
        type=str,
        default="outputs/powerpaint",
    )
    parser.add_argument(
        "--powerpaint-repo-dir",
        type=str,
        default="PowerPaint",
    )
    parser.add_argument(
        "--powerpaint-checkpoint-dir",
        type=str,
        default="checkpoints/PowerPaint-v2-1",
    )
    parser.add_argument(
        "--powerpaint-version",
        type=str,
        default="ppt-v2",
    )
    parser.add_argument(
        "--powerpaint-weight-dtype",
        type=str,
        default="float16",
        choices=[
            "float16",
            "float32",
            "bfloat16",
        ],
    )
    parser.add_argument(
        "--powerpaint-guidance",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--powerpaint-steps",
        type=int,
        default=45,
    )
    parser.add_argument(
        "--powerpaint-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--powerpaint-fitting-degree",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--powerpaint-allow-download",
        action="store_true",
    )
    parser.add_argument(
        "--powerpaint-no-preserve-unmasked",
        action="store_true",
    )
    parser.add_argument(
        "--powerpaint-mask-blur-px",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--powerpaint-task-type",
        type=str,
        default="text-guided",
        choices=[
            "text-guided",
            "shape-guided",
        ],
    )
    parser.add_argument(
        "--powerpaint-negative-prompt",
        type=str,
        default="",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda or cpu.",
    )

    args = parser.parse_args()

    # Keep the original behavior from the pasted wrapper.
    args.force_steps = True
    args.force_edit = True

    ctx = infer_sample_context(args)

    output_dir = os.path.join(
        args.powerpaint_output_root,
        ctx["mode"],
        str(ctx["sample_id"]),
    )

    print("=" * 100)
    print("[StructEdit PowerPaint Pipeline]")
    print(f"mode:          {ctx['mode']}")
    print(
        f"sample_id:     {ctx['sample_id']}"
    )
    print(
        f"image_path:    {ctx['image_path']}"
    )
    print(
        f"sample_dir:    {ctx['sample_dir']}"
    )
    print(
        f"parsed_json:   {ctx['parsed_json']}"
    )
    print(f"output_dir:    {output_dir}")
    print("=" * 100)

    run_structedit_steps(ctx, args)

    result = run_powerpaint_edit(
        ctx,
        args,
    )

    print("[DONE]")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
