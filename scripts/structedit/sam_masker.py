import os
import re
import sys
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List, Any, Tuple

import numpy as np
import torch

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from structedit.config import ensure_dir
except Exception:
    def ensure_dir(path: str):
        if path:
            os.makedirs(path, exist_ok=True)


SAM2_ROOT = "/root/StructEdit/checkpoints/sam2"
SAM2_MODEL_CFG = "sam2_hiera_l.yaml"
SAM2_CHECKPOINT_PATH = "/root/StructEdit/checkpoints/sam2_weights/sam2_hiera_large.pt"


# ============================================================
# Edit type convention
# ============================================================
# add      -> 0
# replace  -> 1
# delete   -> 2
# move     -> 3
# color    -> 4
# material -> 5
EDIT_TYPE_MAP = {
    "add": 0,
    "replace": 1,
    "delete": 2,
    "remove": 2,
    "move": 3,
    "relation": 3,
    "color": 4,
    "material": 5,
}

EDIT_TYPE_ID_TO_NAME = {
    0: "add",
    1: "replace",
    2: "delete",
    3: "move",
    4: "color",
    5: "material",
}


# ============================================================
# Data classes
# ============================================================
@dataclass
class EditUnit:
    target_object: str
    source_object: str
    edit_type: int
    position: int

    operation: str = ""
    attribute: str = ""
    value: str = ""
    position_text: str = ""
    anchor_object: str = ""
    relation: str = ""
    raw_command: str = ""

    source_object_text: str = ""
    target_object_text: str = ""
    anchor_object_text: str = ""

    source_descriptors: List[Dict[str, Any]] = field(default_factory=list)
    target_descriptors: List[Dict[str, Any]] = field(default_factory=list)
    anchor_descriptors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class RelationSpec:
    subject: str
    predicate: str
    object: str
    evidence: str

    def to_dict(self):
        return asdict(self)


@dataclass
class ParsedTask:
    sample_id: str
    image_path: str
    source_prompt: str
    target_prompt: str
    edit_units: List[EditUnit]
    relations: List[RelationSpec]
    source_objects: List[str]
    target_objects: List[str]

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "source_prompt": self.source_prompt,
            "target_prompt": self.target_prompt,
            "edit_units": [u.to_dict() for u in self.edit_units],
            "relations": [r.to_dict() for r in self.relations],
            "source_objects": self.source_objects,
            "target_objects": self.target_objects,
        }


@dataclass
class CandidateDecision:
    unit_index: int
    target_object: str
    source_object: str
    selected_uid: Optional[str]
    bbox_xyxy: Optional[List[float]]

    det_score: float
    relation_score: float
    context_score: float
    total_score: float
    reason: str

    operation: str = ""
    attribute: str = ""
    value: str = ""
    anchor_object: str = ""

    source_object_text: str = ""
    target_object_text: str = ""
    anchor_object_text: str = ""

    source_descriptors: List[Dict[str, Any]] = field(default_factory=list)
    target_descriptors: List[Dict[str, Any]] = field(default_factory=list)
    anchor_descriptors: List[Dict[str, Any]] = field(default_factory=list)

    descriptor_score: float = 0.5
    descriptor_scores: Dict[str, float] = field(default_factory=dict)
    visual_score: float = 0.0
    base_det_score: float = 0.0

    matched_relation: str = ""
    matched_partner_uid: str = ""
    matched_partner_object: str = ""

    # Placement plan. The anchor is localization context and is never an edit mask.
    anchor_uid: Optional[str] = None
    anchor_bbox_xyxy: Optional[List[float]] = None
    placement_bbox_xyxy: Optional[List[float]] = None
    placement_relation: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class VerificationResult:
    target_object: str
    source_object: str
    selected_uid: Optional[str]
    accepted: bool

    final_score: float
    total_score: float
    det_score: float
    relation_score: float
    context_score: float

    clip_score: Optional[float]
    area_ratio: float
    bbox_xyxy_clipped: Optional[List[float]]
    reason: str

    operation: str = ""
    attribute: str = ""
    value: str = ""
    anchor_object: str = ""

    source_object_text: str = ""
    target_object_text: str = ""
    anchor_object_text: str = ""

    visual_score: float = 0.0
    descriptor_score: float = 0.5
    base_det_score: float = 0.0

    source_descriptors: List[Dict[str, Any]] = field(default_factory=list)
    target_descriptors: List[Dict[str, Any]] = field(default_factory=list)
    anchor_descriptors: List[Dict[str, Any]] = field(default_factory=list)

    matched_relation: str = ""
    matched_partner_uid: str = ""
    matched_partner_object: str = ""

    anchor_uid: Optional[str] = None
    anchor_bbox_xyxy: Optional[List[float]] = None
    placement_bbox_xyxy: Optional[List[float]] = None
    placement_relation: str = ""

    def to_dict(self):
        return asdict(self)


# ============================================================
# Basic utils
# ============================================================
def normalize_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def clean_phrase(x: Any) -> str:
    x = normalize_text(x)
    x = x.replace("’", "'")
    x = re.sub(r"^[\s,.;:!?]+|[\s,.;:!?]+$", "", x)
    x = re.sub(r"^(the|a|an)\s+", "", x)
    return x.strip()


def safe_ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)


def make_sample_out_dir(out_dir: str, sample_id: str) -> str:
    sample_dir = os.path.join(out_dir, str(sample_id))
    ensure_dir(sample_dir)
    return sample_dir


def save_json(obj: Any, path: str):
    safe_ensure_parent(path)
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


def unique_keep_order(xs: List[str]) -> List[str]:
    out = []
    for x in xs:
        x = clean_phrase(x)
        if x and x not in out:
            out.append(x)
    return out


def safe_filename(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    return x.strip("_") or "unit"


def get_font(size: int = 15):
    if not PIL_AVAILABLE:
        return None

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ============================================================
# Output path helpers
# ============================================================
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
            return os.path.join("outputs", "PLE_bench", parts[i + 1])

    return os.path.join("outputs", "PLE_bench")


def get_command_sample_dir(parsed_json: str, sample_id: str) -> str:
    parent = os.path.dirname(os.path.abspath(parsed_json))

    if os.path.basename(parent) == str(sample_id):
        return parent

    return make_sample_out_dir(os.path.join("outputs", "command"), sample_id)


# ============================================================
# Parsed task loader
# ============================================================
def _dict_to_edit_unit(d: Dict[str, Any]) -> EditUnit:
    return EditUnit(
        target_object=clean_phrase(d.get("target_object", "")),
        source_object=clean_phrase(d.get("source_object", "")),
        edit_type=int(d.get("edit_type", -1)),
        position=int(d.get("position", -1)),
        operation=clean_phrase(d.get("operation", "")),
        attribute=clean_phrase(d.get("attribute", "")),
        value=clean_phrase(d.get("value", "")),
        position_text=clean_phrase(d.get("position_text", "")),
        anchor_object=clean_phrase(d.get("anchor_object", "")),
        relation=clean_phrase(d.get("relation", "")),
        raw_command=d.get("raw_command", ""),
        source_object_text=clean_phrase(d.get("source_object_text", "")),
        target_object_text=clean_phrase(d.get("target_object_text", "")),
        anchor_object_text=clean_phrase(d.get("anchor_object_text", "")),
        source_descriptors=d.get("source_descriptors", []) or [],
        target_descriptors=d.get("target_descriptors", []) or [],
        anchor_descriptors=d.get("anchor_descriptors", []) or [],
    )


def _dict_to_relation(d: Dict[str, Any]) -> RelationSpec:
    return RelationSpec(
        subject=clean_phrase(d.get("subject", "")),
        predicate=clean_phrase(d.get("predicate", "")),
        object=clean_phrase(d.get("object", "")),
        evidence=d.get("evidence", ""),
    )


def load_parsed_task(path: str) -> ParsedTask:
    data = load_json(path)

    edit_units = [_dict_to_edit_unit(u) for u in data.get("edit_units", [])]
    relations = [_dict_to_relation(r) for r in data.get("relations", [])]

    source_objects = unique_keep_order(data.get("source_objects", []) or [])
    target_objects = unique_keep_order(data.get("target_objects", []) or [])

    if not source_objects:
        source_objects = unique_keep_order([u.source_object for u in edit_units if u.source_object])

    if not target_objects:
        target_objects = unique_keep_order([u.target_object for u in edit_units if u.target_object])

    virtual_add_targets = {
        clean_phrase(u.target_object)
        for u in edit_units
        if (clean_phrase(u.operation) == "add" or int(u.edit_type) == 0) and clean_phrase(u.target_object)
    }

    for r in relations:
        if (
            r.subject
            and r.subject not in ["image", "scene"]
            and r.subject not in virtual_add_targets
            and r.subject not in source_objects
        ):
            source_objects.append(r.subject)

        if (
            r.object
            and r.object not in ["image", "scene"]
            and r.object not in virtual_add_targets
            and r.object not in source_objects
        ):
            source_objects.append(r.object)

    return ParsedTask(
        sample_id=str(data.get("sample_id", "unknown")),
        image_path=data.get("image_path", ""),
        source_prompt=data.get("source_prompt", ""),
        target_prompt=data.get("target_prompt", ""),
        edit_units=edit_units,
        relations=relations,
        source_objects=source_objects,
        target_objects=target_objects,
    )


# ============================================================
# Decision / verification loaders
# ============================================================
def load_decisions(path: str) -> Dict[str, CandidateDecision]:
    data = load_json(path)
    decisions = {}

    for key, x in data.items():
        decisions[key] = CandidateDecision(
            unit_index=int(x.get("unit_index", -1)),
            target_object=clean_phrase(x.get("target_object", "")),
            source_object=clean_phrase(x.get("source_object", "")),
            selected_uid=x.get("selected_uid", None),
            bbox_xyxy=x.get("bbox_xyxy", None),
            det_score=float(x.get("det_score", 0.0)),
            relation_score=float(x.get("relation_score", 0.0)),
            context_score=float(x.get("context_score", 0.0)),
            total_score=float(x.get("total_score", 0.0)),
            reason=x.get("reason", ""),
            operation=clean_phrase(x.get("operation", "")),
            attribute=clean_phrase(x.get("attribute", "")),
            value=clean_phrase(x.get("value", "")),
            anchor_object=clean_phrase(x.get("anchor_object", "")),
            source_object_text=clean_phrase(x.get("source_object_text", x.get("source_object", ""))),
            target_object_text=clean_phrase(x.get("target_object_text", x.get("target_object", ""))),
            anchor_object_text=clean_phrase(x.get("anchor_object_text", x.get("anchor_object", ""))),
            source_descriptors=x.get("source_descriptors", []) or [],
            target_descriptors=x.get("target_descriptors", []) or [],
            anchor_descriptors=x.get("anchor_descriptors", []) or [],
            descriptor_score=float(x.get("descriptor_score", 0.5)),
            descriptor_scores=x.get("descriptor_scores", {}) or {},
            visual_score=float(x.get("visual_score", x.get("det_score", 0.0))),
            base_det_score=float(x.get("base_det_score", x.get("det_score", 0.0))),
            matched_relation=x.get("matched_relation", ""),
            matched_partner_uid=x.get("matched_partner_uid", ""),
            matched_partner_object=clean_phrase(x.get("matched_partner_object", "")),
            anchor_uid=x.get("anchor_uid", None),
            anchor_bbox_xyxy=x.get("anchor_bbox_xyxy", None),
            placement_bbox_xyxy=x.get("placement_bbox_xyxy", None),
            placement_relation=clean_phrase(x.get("placement_relation", x.get("relation", ""))),
        )

    return decisions


def load_verification_results(path: str) -> Dict[str, VerificationResult]:
    data = load_json(path)
    results = {}

    for key, x in data.items():
        results[key] = VerificationResult(
            target_object=clean_phrase(x.get("target_object", "")),
            source_object=clean_phrase(x.get("source_object", "")),
            selected_uid=x.get("selected_uid", None),
            accepted=bool(x.get("accepted", False)),
            final_score=float(x.get("final_score", 0.0)),
            total_score=float(x.get("total_score", 0.0)),
            det_score=float(x.get("det_score", 0.0)),
            relation_score=float(x.get("relation_score", 0.0)),
            context_score=float(x.get("context_score", 0.0)),
            clip_score=x.get("clip_score", None),
            area_ratio=float(x.get("area_ratio", 0.0)),
            bbox_xyxy_clipped=x.get("bbox_xyxy_clipped", None),
            reason=x.get("reason", ""),
            operation=clean_phrase(x.get("operation", "")),
            attribute=clean_phrase(x.get("attribute", "")),
            value=clean_phrase(x.get("value", "")),
            anchor_object=clean_phrase(x.get("anchor_object", "")),
            source_object_text=clean_phrase(x.get("source_object_text", x.get("source_object", ""))),
            target_object_text=clean_phrase(x.get("target_object_text", x.get("target_object", ""))),
            anchor_object_text=clean_phrase(x.get("anchor_object_text", x.get("anchor_object", ""))),
            visual_score=float(x.get("visual_score", x.get("det_score", 0.0))),
            descriptor_score=float(x.get("descriptor_score", 0.5)),
            base_det_score=float(x.get("base_det_score", x.get("det_score", 0.0))),
            source_descriptors=x.get("source_descriptors", []) or [],
            target_descriptors=x.get("target_descriptors", []) or [],
            anchor_descriptors=x.get("anchor_descriptors", []) or [],
            matched_relation=x.get("matched_relation", ""),
            matched_partner_uid=x.get("matched_partner_uid", ""),
            matched_partner_object=clean_phrase(x.get("matched_partner_object", "")),
            anchor_uid=x.get("anchor_uid", None),
            anchor_bbox_xyxy=x.get("anchor_bbox_xyxy", None),
            placement_bbox_xyxy=x.get("placement_bbox_xyxy", None),
            placement_relation=clean_phrase(x.get("placement_relation", "")),
        )

    return results


# ============================================================
# Mask helpers
# ============================================================
def expand_box_xyxy(
    box_xyxy: List[float],
    pixels: int,
    image_width: int,
    image_height: int,
) -> List[float]:
    if pixels <= 0:
        return [float(v) for v in box_xyxy]

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    x1 -= float(pixels)
    y1 -= float(pixels)
    x2 += float(pixels)
    y2 += float(pixels)

    x1 = max(0.0, min(float(image_width - 1), x1))
    y1 = max(0.0, min(float(image_height - 1), y1))
    x2 = max(0.0, min(float(image_width - 1), x2))
    y2 = max(0.0, min(float(image_height - 1), y2))

    return [x1, y1, x2, y2]


def dilate_binary_mask(mask: np.ndarray, pixels: int = 2) -> np.ndarray:
    """
    Expand a binary mask by N pixels using 8-neighbor dilation.

    pixels=2 means the mask grows outward by roughly 2 pixels.
    No scipy/cv2 dependency is required.
    """
    if pixels <= 0:
        return mask.astype(bool)

    m = mask.astype(bool)

    for _ in range(int(pixels)):
        p = np.pad(m, pad_width=1, mode="constant", constant_values=False)

        m = (
            p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] |
            p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
            p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]
        )

    return m.astype(bool)


def rectangle_mask_from_box(
    image_height: int,
    image_width: int,
    box_xyxy: Optional[List[float]],
) -> Optional[np.ndarray]:
    if box_xyxy is None or len(box_xyxy) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = int(max(0, min(image_width - 1, round(x1))))
    y1 = int(max(0, min(image_height - 1, round(y1))))
    x2 = int(max(0, min(image_width, round(x2))))
    y2 = int(max(0, min(image_height, round(y2))))

    if x2 <= x1 or y2 <= y1:
        return None

    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# ============================================================
# SAM2 masker
# ============================================================
class SAM2Masker:
    """
    Final step:
      00 parsed task
      04 context decisions
      05 verification results
      -> 06 SAM2 masks

    Default behavior:
      - only accepted=True targets are segmented
      - final masks are dilated by mask_expand_pixels pixels
    """

    def __init__(
        self,
        device: Optional[str] = None,
        mask_expand_pixels: int = 2,
        box_expand_pixels: int = 0,
        load_model: bool = True,
    ):
        if not PIL_AVAILABLE:
            raise ImportError("Pillow/PIL is required by SAM2 mask generation. Run: pip install pillow")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_expand_pixels = int(mask_expand_pixels)
        self.box_expand_pixels = int(box_expand_pixels)
        self.predictor = None

        if not load_model:
            print("[SAM2] ADD-only job: skip SAM2 model loading; use placement rectangle mask")
            return

        sys.path.insert(0, SAM2_ROOT)

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise ImportError(
                "SAM2 import failed. Please check SAM2_ROOT.\n"
                f"SAM2_ROOT={SAM2_ROOT}\n"
                f"Original error: {repr(e)}"
            )

        if not os.path.exists(SAM2_CHECKPOINT_PATH):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT_PATH}")

        print(f"[SAM2] loading model on {self.device}")
        print(f"[SAM2] root: {SAM2_ROOT}")
        print(f"[SAM2] cfg: {SAM2_MODEL_CFG}")
        print(f"[SAM2] ckpt: {SAM2_CHECKPOINT_PATH}")
        print(f"[SAM2] mask_expand_pixels: {self.mask_expand_pixels}")
        print(f"[SAM2] box_expand_pixels:  {self.box_expand_pixels}")

        sam2_model = build_sam2(
            SAM2_MODEL_CFG,
            SAM2_CHECKPOINT_PATH,
            device=self.device,
        )

        self.predictor = SAM2ImagePredictor(sam2_model)

    def predict_one_mask(
        self,
        image_np: np.ndarray,
        box_xyxy: List[float],
    ) -> np.ndarray:
        box = np.array(box_xyxy, dtype=np.float32)

        masks_pred, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )

        mask = masks_pred[0]

        if mask.ndim > 2:
            mask = np.squeeze(mask)

        mask = mask.astype(bool)

        if self.mask_expand_pixels > 0:
            mask = dilate_binary_mask(mask, pixels=self.mask_expand_pixels)

        return mask.astype(bool)

    def predict_masks(
        self,
        image_path: str,
        decisions: Dict[str, CandidateDecision],
        verifications: Dict[str, VerificationResult],
        out_dir: str,
    ) -> Dict[str, np.ndarray]:
        ensure_dir(out_dir)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_height, image_width = image_np.shape[:2]

        needs_source_segmentation = any(
            ver.accepted
            and key in decisions
            and clean_phrase(ver.operation or decisions[key].operation) != "add"
            for key, ver in verifications.items()
        )
        if needs_source_segmentation:
            if self.predictor is None:
                raise RuntimeError("SAM2 predictor is required for non-ADD operations")
            self.predictor.set_image(image_np)

        masks: Dict[str, np.ndarray] = {}
        meta: Dict[str, Any] = {}

        for decision_key, verification in verifications.items():
            if not verification.accepted:
                print(f"[SAM2] skip {decision_key}: verification rejected")
                continue

            if decision_key not in decisions:
                print(f"[SAM2] skip {decision_key}: no decision found")
                continue

            decision = decisions[decision_key]
            op = clean_phrase(verification.operation or decision.operation)
            placement_box = (
                verification.placement_bbox_xyxy
                if verification.placement_bbox_xyxy is not None
                else decision.placement_bbox_xyxy
            )
            anchor_box = (
                verification.anchor_bbox_xyxy
                if verification.anchor_bbox_xyxy is not None
                else decision.anchor_bbox_xyxy
            )

            target_name = verification.target_object or decision.target_object or decision_key
            source_name = verification.source_object or decision.source_object

            source_mask = None
            placement_mask = rectangle_mask_from_box(
                image_height=image_height,
                image_width=image_width,
                box_xyxy=placement_box,
            )
            box_xyxy = None
            box_xyxy_prompt = None
            box_source = "none"

            if op != "add":
                if verification.bbox_xyxy_clipped is not None:
                    box_xyxy = verification.bbox_xyxy_clipped
                    box_source = "verification.bbox_xyxy_clipped"
                else:
                    box_xyxy = decision.bbox_xyxy
                    box_source = "decision.bbox_xyxy"

                if box_xyxy is None:
                    print(f"[SAM2] skip {decision_key}: no source bbox")
                    continue

                box_xyxy_prompt = expand_box_xyxy(
                    box_xyxy=box_xyxy,
                    pixels=self.box_expand_pixels,
                    image_width=image_width,
                    image_height=image_height,
                )

                print(
                    f"[SAM2] segment source key={decision_key}, target={target_name}, "
                    f"source={source_name}, uid={decision.selected_uid}, "
                    f"box={box_xyxy_prompt}, box_source={box_source}"
                )

                source_mask = self.predict_one_mask(
                    image_np=image_np,
                    box_xyxy=box_xyxy_prompt,
                )

            if op == "add":
                if placement_mask is None:
                    print(f"[SAM2] skip {decision_key}: ADD has no placement bbox")
                    continue
                final_mask = placement_mask
                mask_type = "add_placement_rectangle"
            elif op == "move":
                if source_mask is None:
                    continue
                final_mask = source_mask.copy()
                if placement_mask is not None:
                    final_mask = np.logical_or(final_mask, placement_mask)
                    mask_type = "move_source_sam_plus_destination_rectangle"
                else:
                    mask_type = "move_source_sam_only_missing_destination"
            else:
                if source_mask is None:
                    continue
                final_mask = source_mask
                mask_type = "source_sam"

            masks[decision_key] = final_mask.astype(bool)

            mask_filename = f"{safe_filename(decision_key)}_mask.png"
            mask_path = os.path.join(out_dir, mask_filename)
            Image.fromarray(final_mask.astype(np.uint8) * 255).save(mask_path)

            source_mask_path = None
            if source_mask is not None:
                source_mask_path = os.path.join(
                    out_dir, f"{safe_filename(decision_key)}_source_mask.png"
                )
                Image.fromarray(source_mask.astype(np.uint8) * 255).save(source_mask_path)

            placement_mask_path = None
            if placement_mask is not None:
                placement_mask_path = os.path.join(
                    out_dir, f"{safe_filename(decision_key)}_placement_mask.png"
                )
                Image.fromarray(placement_mask.astype(np.uint8) * 255).save(placement_mask_path)

            meta[decision_key] = {
                "decision_key": decision_key,
                "operation": op,
                "mask_type": mask_type,
                "target_object": target_name,
                "source_object": source_name,
                "selected_uid": decision.selected_uid,
                "box_xyxy_original": box_xyxy,
                "box_xyxy_prompt": box_xyxy_prompt,
                "box_source": box_source,
                "anchor_uid": verification.anchor_uid or decision.anchor_uid,
                "anchor_bbox_xyxy": anchor_box,
                "placement_bbox_xyxy": placement_box,
                "placement_relation": verification.placement_relation or decision.placement_relation,
                "mask_path": mask_path,
                "source_mask_path": source_mask_path,
                "placement_mask_path": placement_mask_path,
                "mask_expand_pixels": self.mask_expand_pixels,
                "box_expand_pixels": self.box_expand_pixels,
                "mask_area_pixels": int(final_mask.sum()),
                "mask_area_ratio": float(final_mask.sum() / max(1, final_mask.shape[0] * final_mask.shape[1])),
                "verification": verification.to_dict(),
                "decision": decision.to_dict(),
            }

        if masks:
            self.save_combined_mask(
                masks=masks,
                out_path=os.path.join(out_dir, "combined_mask.png"),
            )

            self.visualize_masks(
                image_path=image_path,
                masks=masks,
                decisions=decisions,
                verifications=verifications,
                meta=meta,
                out_path=os.path.join(out_dir, "sam2_mask_overlay.jpg"),
            )

        meta_path = os.path.join(out_dir, "06_sam2_mask_meta.json")
        save_json(meta, meta_path)

        return masks

    def save_combined_mask(self, masks: Dict[str, np.ndarray], out_path: str):
        combined = np.zeros_like(next(iter(masks.values())), dtype=np.uint8)

        for mask in masks.values():
            combined = np.maximum(combined, mask.astype(np.uint8))

        Image.fromarray(combined * 255).save(out_path)

    def _text_size(self, draw, text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _fit_text_to_width(self, draw, text, font, max_width):
        if not text:
            return ""

        if self._text_size(draw, text, font)[0] <= max_width:
            return text

        ellipsis = "..."
        result = text

        while result:
            candidate = result + ellipsis
            if self._text_size(draw, candidate, font)[0] <= max_width:
                return candidate
            result = result[:-1]

        return ellipsis

    def _wrap_text(self, draw, text, font, max_width):
        lines = []
        cur = ""

        for ch in text:
            test = cur + ch
            if self._text_size(draw, test, font)[0] <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = ch

        if cur:
            lines.append(cur)

        return lines

    def _append_bottom_text_panel(
        self,
        image,
        title: str,
        lines: List[str],
        font,
        padding: int = 10,
        line_gap: int = 5,
        panel_fill: str = "black",
        text_fill: str = "white",
    ):
        w, h = image.size

        temp = Image.new("RGB", (w, 10), "white")
        temp_draw = ImageDraw.Draw(temp)

        max_text_width = max(10, w - padding * 2)
        wrapped_lines = []

        if title:
            wrapped_lines.extend(self._wrap_text(temp_draw, title, font, max_text_width))
            wrapped_lines.append("")

        for line in lines:
            wrapped_lines.extend(self._wrap_text(temp_draw, line, font, max_text_width))

        if not wrapped_lines:
            return image

        line_heights = []
        for line in wrapped_lines:
            if line == "":
                line_heights.append(8)
            else:
                _, th = self._text_size(temp_draw, line, font)
                line_heights.append(th)

        panel_h = padding * 2 + sum(line_heights) + line_gap * max(0, len(wrapped_lines) - 1)

        new_image = Image.new("RGB", (w, h + panel_h), "white")
        new_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_image)
        draw.rectangle([0, h, w, h + panel_h], fill=panel_fill)

        y = h + padding
        for line, lh in zip(wrapped_lines, line_heights):
            if line:
                draw.text((padding, y), line, fill=text_fill, font=font)
            y += lh + line_gap

        return new_image

    def visualize_masks(
        self,
        image_path: str,
        masks: Dict[str, np.ndarray],
        decisions: Dict[str, CandidateDecision],
        verifications: Dict[str, VerificationResult],
        meta: Dict[str, Any],
        out_path: str,
        alpha: float = 0.45,
    ):
        image = Image.open(image_path).convert("RGB")
        base = np.array(image).astype(np.float32)

        colors = [
            np.array([255, 0, 0], dtype=np.float32),
            np.array([0, 255, 0], dtype=np.float32),
            np.array([0, 0, 255], dtype=np.float32),
            np.array([255, 255, 0], dtype=np.float32),
            np.array([255, 0, 255], dtype=np.float32),
            np.array([0, 255, 255], dtype=np.float32),
        ]

        for idx, (decision_key, mask) in enumerate(masks.items()):
            color = colors[idx % len(colors)]
            mask_bool = mask.astype(bool)
            base[mask_bool] = base[mask_bool] * (1.0 - alpha) + color * alpha

        overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(overlay)
        font = get_font(15)
        small_font = get_font(13)

        w, h = overlay.size

        bottom_lines = [
            "SAM2 mask overlay",
            "colored area = generated mask",
            "white box = source SAM prompt",
            "cyan box = anchor (context only, never masked)",
            "magenta box = destination placement region",
            f"mask_expand_pixels = {self.mask_expand_pixels}",
            f"box_expand_pixels = {self.box_expand_pixels}",
        ]

        for idx, decision_key in enumerate(masks.keys(), start=1):
            decision = decisions.get(decision_key)
            verification = verifications.get(decision_key)
            item_meta = meta.get(decision_key, {})

            if decision is None:
                bottom_lines.append(f"{idx}. {decision_key} | missing decision")
                continue

            box = item_meta.get("box_xyxy_prompt", None)
            box_source = item_meta.get("box_source", "unknown")

            anchor_box = item_meta.get("anchor_bbox_xyxy", None)
            placement_box = item_meta.get("placement_bbox_xyxy", None)

            if anchor_box is not None and len(anchor_box) == 4:
                ax1, ay1, ax2, ay2 = [float(v) for v in anchor_box]
                draw.rectangle([ax1, ay1, ax2, ay2], outline="cyan", width=3)
                draw.text((ax1 + 3, ay1 + 3), "anchor", fill="cyan", font=small_font)

            if placement_box is not None and len(placement_box) == 4:
                px1, py1, px2, py2 = [float(v) for v in placement_box]
                draw.rectangle([px1, py1, px2, py2], outline="magenta", width=3)
                draw.text((px1 + 3, py1 + 3), "destination", fill="magenta", font=small_font)

            if box is not None:
                x1, y1, x2, y2 = box

                x1 = max(0, min(float(x1), w - 1))
                y1 = max(0, min(float(y1), h - 1))
                x2 = max(0, min(float(x2), w - 1))
                y2 = max(0, min(float(y2), h - 1))

                if x2 > x1 and y2 > y1:
                    draw.rectangle([x1, y1, x2, y2], outline="white", width=3)
                else:
                    x1 = y1 = x2 = y2 = 0.0
            else:
                x1 = y1 = x2 = y2 = 0.0

            label_text = (
                verification.target_object
                if verification is not None and verification.target_object
                else decision_key
            )

            inner_label = ""
            if box is not None and x2 > x1 and y2 > y1:
                inner_label = self._fit_text_to_width(
                    draw=draw,
                    text=str(label_text),
                    font=small_font,
                    max_width=max(20, int(x2 - x1 - 8)),
                )

            if inner_label:
                tw, th = self._text_size(draw, inner_label, small_font)

                tag_x1 = x1
                tag_y1 = y1
                tag_x2 = min(w - 1, x1 + tw + 8)
                tag_y2 = min(h - 1, y1 + th + 6)

                draw.rectangle(
                    [tag_x1, tag_y1, tag_x2, tag_y2],
                    fill="black",
                    outline="white",
                )
                draw.text(
                    (tag_x1 + 4, tag_y1 + 3),
                    inner_label,
                    fill="white",
                    font=small_font,
                )

            mask = masks[decision_key]
            mask_area_ratio = float(mask.sum() / max(1, mask.shape[0] * mask.shape[1]))

            bottom_lines.append(
                f"{idx}. key={decision_key} | op={item_meta.get('operation', decision.operation)} | "
                f"target={label_text} <- {decision.source_object} | uid={decision.selected_uid} | "
                f"source_bbox={box} | anchor_bbox={anchor_box} | "
                f"destination_bbox={placement_box} | bbox_source={box_source} | "
                f"mask_area_ratio={mask_area_ratio:.5f}"
            )

        overlay = self._append_bottom_text_panel(
            image=overlay,
            title="SAM2 visualization",
            lines=bottom_lines,
            font=font,
            padding=10,
            line_gap=5,
            panel_fill="black",
            text_fill="white",
        )

        safe_ensure_parent(out_path)
        overlay.save(out_path)


# ============================================================
# Build task + paths
# ============================================================
def build_task_and_sample_dir(args) -> Tuple[ParsedTask, str, str]:
    has_json = bool(args.input_json)
    has_parsed = bool(args.parsed_json)

    if has_json and has_parsed:
        raise ValueError("Use either --input-json or --parsed-json, not both.")

    if not has_json and not has_parsed:
        raise ValueError(
            "Please provide one input mode:\n"
            "  PLE mode:    --input-json annotations.json --idx 0\n"
            "  Parsed mode: --parsed-json outputs/command/63/00_parsed_task.json"
        )

    # --------------------------------------------------------
    # Mode 1: parsed-json from command parser
    # --------------------------------------------------------
    if has_parsed:
        if not os.path.exists(args.parsed_json):
            raise FileNotFoundError(f"Parsed json not found: {args.parsed_json}")

        task = load_parsed_task(args.parsed_json)
        sample_dir = get_command_sample_dir(args.parsed_json, task.sample_id)
        return task, sample_dir, args.parsed_json

    # --------------------------------------------------------
    # Mode 2: PLE original annotation, but read parsed task
    # --------------------------------------------------------
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    records = load_annotations(args.input_json)

    if args.idx < 0 or args.idx >= len(records):
        raise IndexError(f"--idx {args.idx} out of range, total records={len(records)}")

    record = records[args.idx]
    sample_id = str(record.get("id", "unknown"))

    out_dir = get_ple_output_root(args.input_json)
    sample_dir = make_sample_out_dir(out_dir, sample_id)

    parsed_task_path = os.path.join(sample_dir, "00_parsed_task.json")

    if not os.path.exists(parsed_task_path):
        raise FileNotFoundError(
            f"Parsed task json not found: {parsed_task_path}\n"
            f"Please run rule_parser.py first:\n"
            f"python scripts/structedit/rule_parser.py "
            f"--input-json {args.input_json} --idx {args.idx}"
        )

    task = load_parsed_task(parsed_task_path)

    return task, sample_dir, parsed_task_path


# ============================================================
# Run SAM2 masker
# ============================================================
def run_sam2_masker(
    task: ParsedTask,
    sample_dir: str,
    parsed_task_path: str,
    decisions_json: Optional[str] = None,
    verification_json: Optional[str] = None,
    device: Optional[str] = None,
    mask_expand_pixels: int = 2,
    box_expand_pixels: int = 0,
):
    decisions_json = decisions_json or os.path.join(sample_dir, "04_context_decisions.json")
    verification_json = verification_json or os.path.join(sample_dir, "05_verification_results.json")

    if not os.path.exists(decisions_json):
        raise FileNotFoundError(
            f"Cannot find context decisions: {decisions_json}\n"
            "Please run context_checker.py first with the same input."
        )

    if not os.path.exists(verification_json):
        raise FileNotFoundError(
            f"Cannot find verification results: {verification_json}\n"
            "Please run target_validator.py first with the same input."
        )

    decisions = load_decisions(decisions_json)
    verifications = load_verification_results(verification_json)

    accepted_keys = [
        key
        for key, ver in verifications.items()
        if ver.accepted and key in decisions
    ]

    print("=" * 80)
    print("[SAM2] Using parsed task + context decisions + verification results")
    print(f"[SAM2] parsed task:        {parsed_task_path}")
    print(f"[SAM2] decisions_json:     {decisions_json}")
    print(f"[SAM2] verification_json:  {verification_json}")
    print(f"[SAM2] sample_id:          {task.sample_id}")
    print(f"[SAM2] image_path:         {task.image_path}")
    print(f"[SAM2] sample_dir:         {sample_dir}")
    print(f"[SAM2] accepted keys:      {accepted_keys}")
    print(f"[SAM2] mask_expand_pixels: {mask_expand_pixels}")
    print(f"[SAM2] box_expand_pixels:  {box_expand_pixels}")

    mask_dir = os.path.join(sample_dir, "06_sam2_masks")
    ensure_dir(mask_dir)

    if not accepted_keys:
        print("[SAM2] no accepted targets. Skip SAM2.")
        save_json({}, os.path.join(mask_dir, "06_sam2_mask_meta.json"))
        return

    needs_sam_model = any(
        key in decisions
        and clean_phrase(verifications[key].operation or decisions[key].operation) != "add"
        for key in accepted_keys
    )

    masker = SAM2Masker(
        device=device,
        mask_expand_pixels=mask_expand_pixels,
        box_expand_pixels=box_expand_pixels,
        load_model=needs_sam_model,
    )

    masks = masker.predict_masks(
        image_path=task.image_path,
        decisions=decisions,
        verifications=verifications,
        out_dir=mask_dir,
    )

    print("[SAM2] results:")
    for decision_key, mask in masks.items():
        area_ratio = float(mask.sum() / max(1, mask.shape[0] * mask.shape[1]))
        print(
            f"  {decision_key}: mask_area_pixels={int(mask.sum())}, "
            f"mask_area_ratio={area_ratio:.5f}"
        )

    print(f"[SAM2] saved masks to: {mask_dir}")
    print(f"[SAM2] combined mask:  {os.path.join(mask_dir, 'combined_mask.png')}")
    print(f"[SAM2] overlay:        {os.path.join(mask_dir, 'sam2_mask_overlay.jpg')}")
    print(f"[SAM2] meta:           {os.path.join(mask_dir, '06_sam2_mask_meta.json')}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM2 mask generation from previous StructEdit outputs. "
            "This script reads 00_parsed_task.json, 04_context_decisions.json, "
            "and 05_verification_results.json."
        )
    )

    # Mode 1: original PLE annotation, but loads existing parsed task.
    parser.add_argument("--input-json", type=str, default="", help="Original PLE annotation JSON path.")
    parser.add_argument("--idx", type=int, default=0, help="Sample index for --input-json.")

    # Mode 2: parsed task from command parser.
    parser.add_argument("--parsed-json", type=str, default="", help="Parsed task JSON path, e.g. outputs/command/63/00_parsed_task.json.")

    # Optional explicit dependency paths.
    parser.add_argument("--decisions-json", type=str, default="", help="Optional explicit path to 04_context_decisions.json.")
    parser.add_argument("--verification-json", type=str, default="", help="Optional explicit path to 05_verification_results.json.")

    # Mask controls.
    parser.add_argument(
        "--mask-expand-pixels",
        type=int,
        default=4,
        help="Dilate final binary mask by N pixels. Default is 2.",
    )
    parser.add_argument(
        "--box-expand-pixels",
        type=int,
        default=0,
        help="Optionally expand SAM2 bbox prompt by N pixels before prediction. Default is 0.",
    )

    # Device.
    parser.add_argument("--device", type=str, default="", help="Optional device, e.g. cuda or cpu.")

    args = parser.parse_args()

    task, sample_dir, parsed_task_path = build_task_and_sample_dir(args)

    run_sam2_masker(
        task=task,
        sample_dir=sample_dir,
        parsed_task_path=parsed_task_path,
        decisions_json=args.decisions_json or None,
        verification_json=args.verification_json or None,
        device=args.device or None,
        mask_expand_pixels=args.mask_expand_pixels,
        box_expand_pixels=args.box_expand_pixels,
    )


if __name__ == "__main__":
    main()
