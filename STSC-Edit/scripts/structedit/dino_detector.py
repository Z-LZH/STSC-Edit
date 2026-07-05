import os
import re
import sys
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple

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


GROUNDINGDINO_ROOT = "/root/StructEdit/checkpoints/GroundingDINO"
DINO_CONFIG_PATH = "/root/StructEdit/checkpoints/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT_PATH = "/root/StructEdit/checkpoints/GroundingDINO/weights/groundingdino_swint_ogc.pth"


# =========================================================
# Edit type convention
# =========================================================
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


# =========================================================
# Data classes
# =========================================================
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
class DetectionCandidate:
    uid: str
    object_name: str
    phrase: str
    bbox_xyxy: List[float]
    score: float

    caption: str = ""
    object_text: str = ""
    query_object: str = ""
    descriptors: List[Dict[str, Any]] = field(default_factory=list)

    descriptor_score: float = 0.5
    descriptor_scores: Dict[str, float] = field(default_factory=dict)
    base_det_score: float = 0.0

    def to_dict(self):
        return asdict(self)


# =========================================================
# Constants
# =========================================================
COLOR_WORDS = [
    "red", "blue", "green", "black", "white", "yellow", "purple", "pink",
    "orange", "gray", "grey", "brown", "gold", "silver", "cyan", "magenta",
    "beige", "cream", "navy", "teal", "turquoise", "maroon", "violet",
]

MATERIAL_WORDS = [
    "wood", "wooden", "metal", "metallic", "glass", "plastic", "leather",
    "fabric", "stone", "ceramic", "rubber", "paper", "cotton", "silk",
    "marble", "concrete", "steel", "iron", "bronze", "golden", "silver",
]

SIZE_WORDS = [
    "small", "tiny", "little", "large", "big", "huge", "tall", "short",
]

COLOR_RGB = {
    "red": (220, 30, 30),
    "blue": (40, 90, 220),
    "green": (40, 160, 60),
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "yellow": (230, 210, 40),
    "purple": (150, 70, 180),
    "pink": (235, 120, 180),
    "orange": (235, 130, 35),
    "gray": (130, 130, 130),
    "grey": (130, 130, 130),
    "brown": (140, 85, 45),
    "gold": (220, 170, 50),
    "silver": (180, 180, 180),
    "cyan": (40, 200, 220),
    "magenta": (220, 50, 180),
    "beige": (215, 200, 175),
    "cream": (245, 240, 220),
    "navy": (30, 45, 110),
    "teal": (20, 130, 130),
    "turquoise": (64, 224, 208),
    "maroon": (128, 0, 0),
    "violet": (138, 43, 226),
}

SPATIAL_WORDS = [
    "left", "right", "top", "bottom", "upper", "lower",
    "front", "back", "middle", "center", "central",
]


# =========================================================
# Utilities
# =========================================================
def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def clean_phrase(x: str) -> str:
    x = normalize_text(x)
    x = re.sub(r"^[\s,.;:!?]+|[\s,.;:!?]+$", "", x)
    x = re.sub(r"^(the|a|an)\s+", "", x)
    return x.strip()


def safe_json_loads(x: Any, default: Any):
    if x is None:
        return default
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return default
    return default


def safe_ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)


def make_sample_out_dir(out_dir: str, sample_id: str) -> str:
    sample_dir = os.path.join(out_dir, str(sample_id))
    ensure_dir(sample_dir)
    return sample_dir

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

    # fallback
    return os.path.join("outputs", "PLE_bench")

def save_json(obj, path: str):
    safe_ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
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


def get_font(size: int = 15):
    if not PIL_AVAILABLE:
        return None

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


# =========================================================
# Parsed-task loader
# =========================================================
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

    source_objects = [clean_phrase(x) for x in data.get("source_objects", []) if clean_phrase(x)]
    target_objects = [clean_phrase(x) for x in data.get("target_objects", []) if clean_phrase(x)]

    # fallback: if source_objects / target_objects missing, rebuild from edit_units
    if not source_objects:
        srcs = []
        for u in edit_units:
            if u.source_object and u.source_object not in srcs:
                srcs.append(u.source_object)
            if u.anchor_object and u.anchor_object not in ["image", "scene"] and u.anchor_object not in srcs:
                srcs.append(u.anchor_object)
        source_objects = srcs

    if not target_objects:
        tgts = []
        for u in edit_units:
            if u.target_object and u.target_object not in tgts:
                tgts.append(u.target_object)
        target_objects = tgts

    # relation nodes should also be available as source-side detection objects
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


# =========================================================
# PLE parser
# =========================================================
def parse_annotation_record(record: Dict[str, Any]) -> ParsedTask:
    """
    Parse original PLE annotation record.

    Important fix:
    - edit_action in PLE can be a JSON string, not a dict.
    """
    sample_id = str(record.get("id", "unknown"))
    image_path = record["image"]
    source_prompt = record.get("source_prompt", "")
    target_prompt = record.get("target_prompt", "")

    edit_action = safe_json_loads(record.get("edit_action"), {})
    if not isinstance(edit_action, dict):
        edit_action = {}

    edit_units = []
    source_objects = []
    target_objects = []

    for target_object, info in edit_action.items():
        if not isinstance(info, dict):
            continue

        target_object = clean_phrase(target_object)
        source_object = clean_phrase(info.get("action", target_object))
        edit_type = int(info.get("edit_type", -1))

        unit = EditUnit(
            target_object=target_object,
            source_object=source_object,
            edit_type=edit_type,
            position=int(info.get("position", -1)),
            operation=EDIT_TYPE_ID_TO_NAME.get(edit_type, ""),
            source_object_text=source_object,
            target_object_text=target_object,
        )
        edit_units.append(unit)

        if source_object and source_object not in source_objects:
            source_objects.append(source_object)
        if target_object and target_object not in target_objects:
            target_objects.append(target_object)

    return ParsedTask(
        sample_id=sample_id,
        image_path=image_path,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        edit_units=edit_units,
        relations=[],
        source_objects=source_objects,
        target_objects=target_objects,
    )


# =========================================================
# Descriptor-aware detection target builder
# =========================================================
def _get_task_value(task: Any, key: str, default=None):
    if isinstance(task, dict):
        return task.get(key, default)
    return getattr(task, key, default)


def _get_unit_value(unit: Any, key: str, default=None):
    if isinstance(unit, dict):
        return unit.get(key, default)
    return getattr(unit, key, default)


def get_detection_targets(task: Any) -> List[Dict[str, Any]]:
    """
    Build descriptor-aware GroundingDINO targets.

    Rules:
    - delete / remove / replace / move / color / material -> detect source_object
    - add / move with anchor_object -> also detect anchor_object
    - relation subject/object -> also detect both nodes
    - fallback -> use source_objects

    Examples:
      change left person's shirt to blue
        edit unit detects: shirt
        relation detects: person + shirt
        final targets: shirt(with descriptor), person

      a cat wearing a pink hat
        relations detect: cat + hat
    """
    targets: List[Dict[str, Any]] = []
    seen = {}

    def add_target(base_object: str, object_text: str = "", descriptors=None, unit_index=None):
        base_object = clean_phrase(base_object)
        object_text = clean_phrase(object_text) or base_object
        descriptors = descriptors or []

        if not base_object:
            return

        if base_object in ["image", "scene"]:
            return

        key = json.dumps(
            {
                "query_object": base_object,
                "object_text": object_text,
                "descriptors": descriptors,
            },
            sort_keys=True,
            ensure_ascii=False,
        )

        if key not in seen:
            seen[key] = len(targets)
            targets.append(
                {
                    "query_object": base_object,
                    "object_text": object_text,
                    "descriptors": descriptors,
                    "unit_indices": [],
                }
            )

        if unit_index is not None:
            targets[seen[key]]["unit_indices"].append(unit_index)

    edit_units = _get_task_value(task, "edit_units", []) or []

    # --------------------------------------------------------
    # 1. Targets from edit units
    # --------------------------------------------------------
    for i, unit in enumerate(edit_units):
        op = clean_phrase(_get_unit_value(unit, "operation", ""))
        edit_type = int(_get_unit_value(unit, "edit_type", -1))

        source_object = clean_phrase(_get_unit_value(unit, "source_object", ""))
        source_text = clean_phrase(_get_unit_value(unit, "source_object_text", source_object))
        source_desc = _get_unit_value(unit, "source_descriptors", []) or []

        anchor_object = clean_phrase(_get_unit_value(unit, "anchor_object", ""))
        anchor_text = clean_phrase(_get_unit_value(unit, "anchor_object_text", anchor_object))
        anchor_desc = _get_unit_value(unit, "anchor_descriptors", []) or []

        # detect the source object for these edit types
        if op in ["delete", "remove", "replace", "move", "color", "material"] or edit_type in [1, 2, 3, 4, 5]:
            add_target(source_object, source_text, source_desc, i)

        # add / move may need anchor object
        if (op in ["add", "move"] or edit_type in [0, 3]) and anchor_object:
            add_target(anchor_object, anchor_text, anchor_desc, i)

    # --------------------------------------------------------
    # 2. Targets from parsed relations
    #    person --wearing--> shirt
    #    cat --wearing--> hat
    # --------------------------------------------------------
    relations = _get_task_value(task, "relations", []) or []

    # ADD targets are virtual: they are absent from the source image.
    # Their relations are still useful later to compute an insertion region.
    virtual_add_targets = set()
    for unit in edit_units:
        op = clean_phrase(_get_unit_value(unit, "operation", ""))
        edit_type = int(_get_unit_value(unit, "edit_type", -1))
        target_object = clean_phrase(_get_unit_value(unit, "target_object", ""))
        if (op == "add" or edit_type == 0) and target_object:
            virtual_add_targets.add(target_object)

    for rel in relations:
        if isinstance(rel, dict):
            subject = clean_phrase(rel.get("subject", ""))
            obj = clean_phrase(rel.get("object", ""))
            evidence = clean_phrase(rel.get("evidence", ""))
        else:
            subject = clean_phrase(getattr(rel, "subject", ""))
            obj = clean_phrase(getattr(rel, "object", ""))
            evidence = clean_phrase(getattr(rel, "evidence", ""))

        # For relation nodes we usually do not have descriptor info here.
        # Detection target should be the clean base object.
        if subject and subject not in virtual_add_targets:
            add_target(subject, subject, [], None)

        if obj and obj not in virtual_add_targets:
            add_target(obj, obj, [], None)

    # --------------------------------------------------------
    # 3. Fallback from source_objects
    # --------------------------------------------------------
    if not targets:
        for obj in _get_task_value(task, "source_objects", []) or []:
            obj = clean_phrase(obj)
            if obj:
                add_target(obj, obj, [], None)

    return targets

# =========================================================
# Descriptor scoring
# =========================================================
def descriptor_to_text(descriptors: List[Dict[str, Any]]) -> str:
    parts = []
    for d in descriptors or []:
        value = d.get("value", "")
        obj = d.get("object", "")
        if obj:
            parts.append(f"{value} {obj}".strip())
        elif value:
            parts.append(str(value))
    return ", ".join([p for p in parts if p])


def get_descriptor_score_for_box(
    image_source: np.ndarray,
    bbox_xyxy: List[float],
    descriptors: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Score how well a detected box matches descriptors.
    Supports:
    - position / spatial
    - color
    - size
    - relation object color hints
    """
    if not descriptors:
        return 0.5, {}

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height - 1))

    if x2 <= x1 or y2 <= y1:
        return 0.0, {"invalid_bbox": 0.0}

    cx = (x1 + x2) / 2.0 / max(1.0, float(image_width))
    cy = (y1 + y2) / 2.0 / max(1.0, float(image_height))
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(image_width * image_height))

    crop = image_source[int(y1):int(y2), int(x1):int(x2)]
    crop_rgb = None
    if crop is not None and crop.size > 0:
        crop_rgb = crop.astype(np.float32) / 255.0

    scores: Dict[str, float] = {}

    for i, d in enumerate(descriptors):
        dtype = str(d.get("type", "")).lower()
        value = str(d.get("value", "")).lower()
        key = f"{i}_{dtype}_{value}".strip("_")
        score = 0.5

        # support both parser formats:
        # - {"type":"position","value":"left"}
        # - {"type":"spatial","value":"left"}
        if dtype in ["spatial", "position"]:
            if value in ["left", "leftmost"]:
                score = 1.0 - cx
            elif value in ["right", "rightmost"]:
                score = cx
            elif value in ["top", "upper", "topmost"]:
                score = 1.0 - cy
            elif value in ["bottom", "lower", "bottommost"]:
                score = cy
            elif value in ["middle", "center", "central"]:
                dist = float(np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2))
                score = max(0.0, 1.0 - dist / 0.7071)
            else:
                score = 0.5

        elif dtype == "size":
            if value in ["small", "tiny", "little", "short"]:
                score = max(0.0, 1.0 - min(1.0, area_ratio * 12.0))
            elif value in ["large", "big", "huge", "tall"]:
                score = min(1.0, area_ratio * 8.0)
            else:
                score = 0.5

        elif dtype == "color":
            color = value
            if color in COLOR_RGB and crop_rgb is not None and crop_rgb.size > 0:
                target = np.array(COLOR_RGB[color], dtype=np.float32) / 255.0
                dist = np.linalg.norm(crop_rgb - target.reshape(1, 1, 3), axis=2) / np.sqrt(3.0)
                closest = np.quantile(dist, 0.20)
                score = float(np.clip(1.0 - closest, 0.0, 1.0))
            else:
                score = 0.5

        elif dtype == "relation":
            obj_text = str(d.get("object", "")).lower()
            color_hits = [c for c in COLOR_RGB.keys() if re.search(rf"\b{re.escape(c)}\b", obj_text)]
            if color_hits and crop_rgb is not None and crop_rgb.size > 0:
                color = color_hits[0]
                target = np.array(COLOR_RGB[color], dtype=np.float32) / 255.0
                dist = np.linalg.norm(crop_rgb - target.reshape(1, 1, 3), axis=2) / np.sqrt(3.0)
                closest = np.quantile(dist, 0.20)
                score = float(np.clip(1.0 - closest, 0.0, 1.0))
            else:
                score = 0.5

        elif dtype == "material":
            score = 0.5  # lightweight neutral

        else:
            score = 0.5

        scores[key] = float(np.clip(score, 0.0, 1.0))

    return float(np.mean(list(scores.values()))) if scores else 0.5, scores

def has_spatial_descriptor(descriptors: List[Dict[str, Any]]) -> bool:
    for d in descriptors or []:
        dtype = str(d.get("type", "")).lower()
        value = str(d.get("value", "")).lower()

        if dtype in ["spatial", "position"] and value in [
            "left", "leftmost",
            "right", "rightmost",
            "top", "upper", "topmost",
            "bottom", "lower", "bottommost",
            "middle", "center", "central",
        ]:
            return True

    return False


def combine_detection_and_descriptor_score(
    base_det_score: float,
    descriptor_score: float,
    descriptors: List[Dict[str, Any]],
) -> float:
    """
    If the command has spatial descriptors like top/left/right/bottom,
    trust descriptor_score more than raw GroundingDINO score.

    Example:
      "top knife" should prefer the highest knife in the image,
      even if another knife has a higher DINO confidence.
    """
    base_det_score = float(base_det_score)
    descriptor_score = float(descriptor_score)

    if not descriptors:
        return base_det_score

    if has_spatial_descriptor(descriptors):
        # Strongly prefer spatial match.
        return 0.20 * base_det_score + 0.80 * descriptor_score

    # For color/material/size descriptors, still consider DINO confidence more.
    return 0.55 * base_det_score + 0.45 * descriptor_score

def normalize_detection_target(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        query_object = clean_phrase(x.get("query_object", x.get("base_object", x.get("object_name", ""))))
        object_text = clean_phrase(x.get("object_text", query_object))
        descriptors = x.get("descriptors", []) or []
        unit_indices = x.get("unit_indices", [])
        return {
            "query_object": query_object,
            "object_text": object_text or query_object,
            "descriptors": descriptors,
            "unit_indices": unit_indices,
        }

    x = clean_phrase(x)
    return {
        "query_object": x,
        "object_text": x,
        "descriptors": [],
        "unit_indices": [],
    }


# =========================================================
# GroundingDINO detector
# =========================================================
class DINODetector:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        sys.path.insert(0, GROUNDINGDINO_ROOT)

        try:
            from groundingdino.util.inference import load_model, load_image, predict
            from groundingdino.util import box_ops
        except Exception as e:
            raise ImportError(
                "GroundingDINO import failed. Please check GroundingDINO paths.\n"
                f"GROUNDINGDINO_ROOT={GROUNDINGDINO_ROOT}\n"
                f"Original error: {repr(e)}"
            )

        if not os.path.exists(DINO_CONFIG_PATH):
            raise FileNotFoundError(f"DINO config not found: {DINO_CONFIG_PATH}")

        if not os.path.exists(DINO_CHECKPOINT_PATH):
            raise FileNotFoundError(f"DINO checkpoint not found: {DINO_CHECKPOINT_PATH}")

        self.load_image_func = load_image
        self.predict_func = predict
        self.box_ops = box_ops

        self.model = load_model(DINO_CONFIG_PATH, DINO_CHECKPOINT_PATH)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect_objects(
        self,
        image_path: str,
        object_names: List[Any],
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        max_per_object: int = 8,
    ) -> List[DetectionCandidate]:

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_source, image_tensor = self.load_image_func(image_path)
        h, w = image_source.shape[:2]

        detection_targets = []
        seen = set()
        for x in object_names:
            t = normalize_detection_target(x)
            if not t["query_object"]:
                continue
            key = json.dumps(t, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                detection_targets.append(t)

        all_candidates = []

        for target in detection_targets:
            obj = target["query_object"]
            object_text = target.get("object_text", obj)
            descriptors = target.get("descriptors", []) or []

            obj_candidates = []
            caption = obj if obj.endswith(".") else obj + "."

            desc_text = descriptor_to_text(descriptors)
            print(
                f"[DINO] detecting base object: {obj}, caption: {caption}, "
                f"object_text={object_text}, descriptors={desc_text}"
            )

            boxes, logits, phrases = self.predict_func(
                model=self.model,
                image=image_tensor,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )

            if boxes is None or len(boxes) == 0:
                print(f"[DINO] no candidate found for object={obj}, caption={caption}")
                continue

            boxes_xyxy = self.box_ops.box_cxcywh_to_xyxy(boxes)
            scale = torch.tensor([w, h, w, h], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
            boxes_xyxy = boxes_xyxy * scale

            base_scores = logits.detach().cpu().float().numpy().tolist()
            boxes_xyxy = boxes_xyxy.detach().cpu().float().numpy().tolist()
            phrases = list(phrases)

            adjusted_scores = []
            descriptor_scores = []
            descriptor_score_maps = []

            for box, base_score in zip(boxes_xyxy, base_scores):
                desc_score, desc_score_map = get_descriptor_score_for_box(
                    image_source=image_source,
                    bbox_xyxy=box,
                    descriptors=descriptors,
                    image_width=w,
                    image_height=h,
                )

                descriptor_scores.append(desc_score)
                descriptor_score_maps.append(desc_score_map)

                adjusted = combine_detection_and_descriptor_score(
                    base_det_score=float(base_score),
                    descriptor_score=float(desc_score),
                    descriptors=descriptors,
                )

                adjusted_scores.append(float(adjusted))
                
            order = np.argsort(adjusted_scores)[::-1][:max_per_object]

            for idx in order:
                uid = f"{obj.replace(' ', '_')}_{len(all_candidates) + len(obj_candidates):04d}"

                obj_candidates.append(
                    DetectionCandidate(
                        uid=uid,
                        object_name=obj,
                        phrase=str(phrases[idx]) if idx < len(phrases) else obj,
                        bbox_xyxy=[float(v) for v in boxes_xyxy[idx]],
                        score=float(adjusted_scores[idx]),
                        caption=obj,
                        object_text=object_text,
                        query_object=obj,
                        descriptors=descriptors,
                        descriptor_score=float(descriptor_scores[idx]),
                        descriptor_scores=descriptor_score_maps[idx],
                        base_det_score=float(base_scores[idx]),
                    )
                )

            if not obj_candidates:
                print(f"[DINO] no candidate found for object: {obj}")
                continue

            all_candidates.extend(obj_candidates[:max_per_object])

        return all_candidates

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

    def visualize_candidates(
        self,
        image_path: str,
        candidates: List[DetectionCandidate],
        out_path: str,
        title: str = "GroundingDINO candidates",
    ):
        if not PIL_AVAILABLE:
            txt_path = os.path.splitext(out_path)[0] + ".txt"
            lines = [title]
            for idx, cand in enumerate(candidates, start=1):
                lines.append(
                    f"{idx}. {cand.object_name} | score={cand.score:.2f} | "
                    f"caption={cand.caption} | phrase={cand.phrase} | bbox={cand.bbox_xyxy}"
                )
            safe_ensure_parent(txt_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"[DINO] PIL is not installed. Saved text visualization: {txt_path}")
            return

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = get_font(15)
        small_font = get_font(13)

        w, h = image.size
        bottom_lines = []

        for idx, cand in enumerate(candidates, start=1):
            x1, y1, x2, y2 = cand.bbox_xyxy

            x1 = max(0, min(float(x1), w - 1))
            y1 = max(0, min(float(y1), h - 1))
            x2 = max(0, min(float(x2), w - 1))
            y2 = max(0, min(float(y2), h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

            object_name = str(cand.object_name)
            score = float(cand.score)

            inner_label = self._fit_text_to_width(
                draw=draw,
                text=object_name,
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
                    outline="yellow",
                )
                draw.text(
                    (tag_x1 + 4, tag_y1 + 3),
                    inner_label,
                    fill="white",
                    font=small_font,
                )

            bottom_lines.append(
                f"{idx}. {object_name} | score={score:.2f} | "
                f"base={cand.base_det_score:.2f} | desc={cand.descriptor_score:.2f} | "
                f"bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) | "
                f"text={cand.object_text}"
            )

        image = self._append_bottom_text_panel(
            image=image,
            title=title,
            lines=bottom_lines,
            font=font,
            padding=10,
            line_gap=5,
            panel_fill="black",
            text_fill="white",
        )

        safe_ensure_parent(out_path)
        image.save(out_path)


# =========================================================
# Save helpers
# =========================================================
def save_candidates(candidates: List[DetectionCandidate], path: str):
    save_json([c.to_dict() for c in candidates], path)


def save_parsed_task(task: ParsedTask, sample_dir: str):
    save_json(task.to_dict(), os.path.join(sample_dir, "00_parsed_task.json"))


# =========================================================
# Main run
# =========================================================
def run_task(
    task: ParsedTask,
    sample_dir: str,
    box_threshold: float,
    text_threshold: float,
    max_per_object: int,
    device: Optional[str] = None,
):
    save_parsed_task(task, sample_dir)

    detection_targets = get_detection_targets(task)

    print("=" * 80)
    print("[DINO] Using parsed task")
    print(f"[DINO] sample_id: {task.sample_id}")
    print(f"[DINO] image_path: {task.image_path}")
    print(f"[DINO] target_prompt: {task.target_prompt}")
    print(f"[DINO] source_objects: {task.source_objects}")
    print(f"[DINO] target_objects: {task.target_objects}")
    print(f"[DINO] detection targets: {detection_targets}")
    print(f"[DINO] saved parsed task: {os.path.join(sample_dir, '00_parsed_task.json')}")
    print("=" * 80)

    if not detection_targets:
        print("[DINO] no detection target found. Skip GroundingDINO.")
        candidates = []
        save_candidates(candidates, os.path.join(sample_dir, "01_dino_candidates.json"))
        print(f"[DINO] num candidates: {len(candidates)}")
        print(f"[DINO] saved json: {os.path.join(sample_dir, '01_dino_candidates.json')}")
        return

    detector = DINODetector(device=device)

    candidates = detector.detect_objects(
        image_path=task.image_path,
        object_names=detection_targets,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        max_per_object=max_per_object,
    )

    candidates_path = os.path.join(sample_dir, "01_dino_candidates.json")
    vis_path = os.path.join(sample_dir, "01_dino_candidates.jpg")

    save_candidates(candidates, candidates_path)
    detector.visualize_candidates(
        image_path=task.image_path,
        candidates=candidates,
        out_path=vis_path,
        title="GroundingDINO candidates from parsed task",
    )

    print(f"[DINO] num candidates: {len(candidates)}")
    print(f"[DINO] saved json: {candidates_path}")
    if PIL_AVAILABLE:
        print(f"[DINO] saved visualization: {vis_path}")
    else:
        print(f"[DINO] saved visualization text: {os.path.splitext(vis_path)[0] + '.txt'}")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run GroundingDINO on either:\n"
            "1) original PLE annotation JSON via --input-json --idx\n"
            "2) parsed task JSON from rule parser via --parsed-json"
        )
    )

    # mode 1: original PLE annotation
    parser.add_argument("--input-json", type=str, default="", help="Original PLE annotation JSON path.")
    parser.add_argument("--idx", type=int, default=0, help="Sample index for --input-json.")

    # mode 2: parsed_task.json from script 1
    parser.add_argument("--parsed-json", type=str, default="", help="Parsed task json path from rule parser.")

    # output / detector config
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-per-object", type=int, default=8)
    parser.add_argument("--device", type=str, default="", help="Optional device, e.g. cuda or cpu.")

    args = parser.parse_args()

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

    if has_parsed:
        if not os.path.exists(args.parsed_json):
            raise FileNotFoundError(f"Parsed json not found: {args.parsed_json}")

        task = load_parsed_task(args.parsed_json)

        # CMD / parsed-json mode output:
        # outputs/command/<id>/
        out_dir = os.path.join("outputs", "command")
        sample_dir = make_sample_out_dir(out_dir, task.sample_id)

    else:
        if not os.path.exists(args.input_json):
            raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

        records = load_annotations(args.input_json)

        if args.idx < 0 or args.idx >= len(records):
            raise IndexError(f"--idx {args.idx} out of range, total records={len(records)}")

        record = records[args.idx]
        sample_id = str(record.get("id", "unknown"))

        # PLE mode output:
        # outputs/PLE_bench/0_random_140/<id>/
        out_dir = get_ple_output_root(args.input_json)
        sample_dir = make_sample_out_dir(out_dir, sample_id)

        parsed_task_path = os.path.join(sample_dir, "00_parsed_task.json")

        # IMPORTANT:
        # PLE mode must read the parsed json generated by rule_parser.py.
        # Do NOT parse annotation again here, otherwise relations like
        # cat --wearing--> hat will be lost.
        if not os.path.exists(parsed_task_path):
            raise FileNotFoundError(
                f"Parsed task json not found: {parsed_task_path}\n"
                f"Please run rule_parser.py first:\n"
                f"python scripts/structedit/rule_parser.py "
                f"--input-json {args.input_json} --idx {args.idx}"
            )

        print(f"[DINO] loading parsed task from rule_parser output: {parsed_task_path}")
        task = load_parsed_task(parsed_task_path)

    run_task(
        task=task,
        sample_dir=sample_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_per_object=args.max_per_object,
        device=args.device or None,
    )


if __name__ == "__main__":
    main()
