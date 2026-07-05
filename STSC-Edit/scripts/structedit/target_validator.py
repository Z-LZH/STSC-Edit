import os
import re
import sys
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List, Any, Tuple

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


@dataclass
class GraphNode:
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
    visual_score: float
    descriptor_score: float
    base_det_score: float

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


def get_font(size: int = 15):
    if not PIL_AVAILABLE:
        return None

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def safe_filename(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    return x.strip("_") or "unit"


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
# Candidate / decision loaders
# ============================================================
def load_candidates(path: str) -> List[DetectionCandidate]:
    data = load_json(path)

    candidates = []
    for x in data:
        candidates.append(
            DetectionCandidate(
                uid=x.get("uid", ""),
                object_name=clean_phrase(x.get("object_name", "")),
                phrase=x.get("phrase", ""),
                bbox_xyxy=x.get("bbox_xyxy", []),
                score=float(x.get("score", 0.0)),
                caption=x.get("caption", x.get("caption_used", "")),
                object_text=x.get("object_text", x.get("object_name", "")),
                query_object=x.get("query_object", x.get("object_name", "")),
                descriptors=x.get("descriptors", []) or [],
                descriptor_score=float(x.get("descriptor_score", 0.5)),
                descriptor_scores=x.get("descriptor_scores", {}) or {},
                base_det_score=float(x.get("base_det_score", x.get("score", 0.0))),
            )
        )

    return candidates


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


def save_verification_results(results: Dict[str, VerificationResult], path: str):
    save_json({k: v.to_dict() for k, v in results.items()}, path)


def load_verification_results(path: str) -> Dict[str, VerificationResult]:
    data = load_json(path)
    return {k: VerificationResult(**v) for k, v in data.items()}


# ============================================================
# Relation graph
# ============================================================
class RelationGraph:
    def __init__(self, nodes: Dict[str, GraphNode]):
        self.nodes = nodes

    @classmethod
    def from_candidates(cls, candidates: List[DetectionCandidate]):
        nodes = {}

        for cand in candidates:
            if not cand.uid:
                continue

            nodes[cand.uid] = GraphNode(
                uid=cand.uid,
                object_name=clean_phrase(cand.object_name),
                phrase=cand.phrase,
                bbox_xyxy=cand.bbox_xyxy,
                score=float(cand.score),
                caption=cand.caption,
                object_text=cand.object_text,
                query_object=clean_phrase(cand.query_object or cand.object_name),
                descriptors=cand.descriptors or [],
                descriptor_score=float(cand.descriptor_score),
                descriptor_scores=cand.descriptor_scores or {},
                base_det_score=float(cand.base_det_score),
            )

        return cls(nodes)

    def nodes_by_object(self, object_name: str) -> List[GraphNode]:
        object_name = clean_phrase(object_name)

        return [
            n for n in self.nodes.values()
            if clean_phrase(n.object_name) == object_name
            or clean_phrase(n.query_object) == object_name
        ]


# ============================================================
# Target validator
# ============================================================
class TargetValidator:
    """
    Sixth step:
      00 parsed task
      01 DINO candidates
      04 context decisions
      -> 05 verification results

    Only accepted=True targets should enter the next SAM2 step.
    """

    def __init__(
        self,
        accept_threshold: float = 0.45,
        min_det_score: float = 0.15,
        min_relation_score: float = 0.0,
        min_context_score: float = 0.0,
        min_area_ratio: float = 0.0002,
        max_area_ratio: float = 0.98,
        enable_clip: bool = False,
        clip_model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.accept_threshold = float(accept_threshold)
        self.min_det_score = float(min_det_score)
        self.min_relation_score = float(min_relation_score)
        self.min_context_score = float(min_context_score)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)

        self.enable_clip = bool(enable_clip)
        self.clip_model_name_or_path = clip_model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model = None
        self.clip_processor = None

        if self.enable_clip:
            self.load_clip()

    def load_clip(self):
        if not PIL_AVAILABLE:
            print("[WARN] PIL is not installed. Disable CLIP.")
            self.enable_clip = False
            return

        try:
            from transformers import CLIPModel, CLIPProcessor

            name = self.clip_model_name_or_path or "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(name).to(self.device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(name)

            print(f"[Target Validator] CLIP loaded: {name}")

        except Exception as e:
            print(f"[WARN] CLIP load failed. Disable CLIP. Error={repr(e)}")
            self.enable_clip = False
            self.clip_model = None
            self.clip_processor = None

    def clip_bbox(self, image_path: str, bbox: List[float]) -> Optional[List[float]]:
        if not PIL_AVAILABLE:
            return None

        if not os.path.exists(image_path):
            return None

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox]

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        x1 = max(0.0, min(float(w), x1))
        y1 = max(0.0, min(float(h), y1))
        x2 = max(0.0, min(float(w), x2))
        y2 = max(0.0, min(float(h), y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    def bbox_area_ratio(self, image_path: str, bbox: List[float]) -> float:
        if not PIL_AVAILABLE:
            return 0.0

        if not os.path.exists(image_path):
            return 0.0

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        clipped = self.clip_bbox(image_path, bbox)
        if clipped is None:
            return 0.0

        x1, y1, x2, y2 = clipped
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return float(area / max(1.0, w * h))

    @torch.no_grad()
    def clip_crop_text_score(self, image_path: str, bbox: List[float], text: str) -> Optional[float]:
        if not self.enable_clip or self.clip_model is None or self.clip_processor is None:
            return None

        if not PIL_AVAILABLE:
            return None

        if not os.path.exists(image_path):
            return None

        image = Image.open(image_path).convert("RGB")
        clipped = self.clip_bbox(image_path, bbox)

        if clipped is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in clipped]
        crop = image.crop((x1, y1, x2, y2))

        inputs = self.clip_processor(
            text=[text],
            images=crop,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image

        return float(torch.sigmoid(logits[0, 0]).item())

    def compute_final_score(
        self,
        decision: CandidateDecision,
        clip_score: Optional[float] = None,
    ) -> float:
        """
        Scores are already fused in 04_context_decisions.json.
        Use context-enhanced total_score as the main score.
        """
        total = float(decision.total_score)
        det = float(decision.det_score)
        rel = float(decision.relation_score)
        ctx = float(decision.context_score)
        visual = float(decision.visual_score)

        final_score = (
            0.45 * total
            + 0.20 * det
            + 0.15 * visual
            + 0.10 * rel
            + 0.10 * ctx
        )

        if clip_score is not None:
            final_score = 0.85 * final_score + 0.15 * float(clip_score)

        return float(final_score)

    def is_add_without_existing_source(self, decision: CandidateDecision) -> bool:
        op = clean_phrase(decision.operation)
        source = clean_phrase(decision.source_object)
        return op == "add" and not source

    def verify(
        self,
        image_path: str,
        decision: CandidateDecision,
    ) -> VerificationResult:
        # ADD has no existing source object. It is valid when the anchor was
        # localized and a destination region was derived from the relation.
        if self.is_add_without_existing_source(decision):
            placement_ok = (
                decision.anchor_uid is not None
                and decision.anchor_bbox_xyxy is not None
                and decision.placement_bbox_xyxy is not None
            )
            return VerificationResult(
                target_object=decision.target_object,
                source_object=decision.source_object,
                selected_uid=None,
                accepted=bool(placement_ok),
                final_score=1.0 if placement_ok else 0.0,
                total_score=float(decision.total_score),
                det_score=float(decision.det_score),
                relation_score=float(decision.relation_score),
                context_score=float(decision.context_score),
                visual_score=float(decision.visual_score),
                descriptor_score=float(decision.descriptor_score),
                base_det_score=float(decision.base_det_score),
                clip_score=None,
                area_ratio=0.0,
                bbox_xyxy_clipped=None,
                reason=("accepted_add_with_anchor_placement_box" if placement_ok else "rejected_add_missing_anchor_or_placement_box"),
                operation=decision.operation,
                attribute=decision.attribute,
                value=decision.value,
                anchor_object=decision.anchor_object,
                source_object_text=decision.source_object_text,
                target_object_text=decision.target_object_text,
                anchor_object_text=decision.anchor_object_text,
                source_descriptors=decision.source_descriptors,
                target_descriptors=decision.target_descriptors,
                anchor_descriptors=decision.anchor_descriptors,
                matched_relation=decision.matched_relation,
                matched_partner_uid=decision.matched_partner_uid,
                matched_partner_object=decision.matched_partner_object,
                anchor_uid=decision.anchor_uid,
                anchor_bbox_xyxy=decision.anchor_bbox_xyxy,
                placement_bbox_xyxy=decision.placement_bbox_xyxy,
                placement_relation=decision.placement_relation,
            )

        if decision.selected_uid is None or decision.bbox_xyxy is None:
            return VerificationResult(
                target_object=decision.target_object,
                source_object=decision.source_object,
                selected_uid=None,
                accepted=False,
                final_score=0.0,
                total_score=float(decision.total_score),
                det_score=float(decision.det_score),
                relation_score=float(decision.relation_score),
                context_score=float(decision.context_score),
                visual_score=float(decision.visual_score),
                descriptor_score=float(decision.descriptor_score),
                base_det_score=float(decision.base_det_score),
                clip_score=None,
                area_ratio=0.0,
                bbox_xyxy_clipped=None,
                reason="rejected_no_selected_candidate",
                operation=decision.operation,
                attribute=decision.attribute,
                value=decision.value,
                anchor_object=decision.anchor_object,
                source_object_text=decision.source_object_text,
                target_object_text=decision.target_object_text,
                anchor_object_text=decision.anchor_object_text,
                source_descriptors=decision.source_descriptors,
                target_descriptors=decision.target_descriptors,
                anchor_descriptors=decision.anchor_descriptors,
                matched_relation=decision.matched_relation,
                matched_partner_uid=decision.matched_partner_uid,
                matched_partner_object=decision.matched_partner_object,
                anchor_uid=decision.anchor_uid,
                anchor_bbox_xyxy=decision.anchor_bbox_xyxy,
                placement_bbox_xyxy=decision.placement_bbox_xyxy,
                placement_relation=decision.placement_relation,
            )

        clipped_bbox = self.clip_bbox(image_path, decision.bbox_xyxy)
        area_ratio = self.bbox_area_ratio(image_path, decision.bbox_xyxy)

        clip_text = (
            decision.source_object_text
            or decision.source_object
            or decision.target_object_text
            or decision.target_object
        )

        clip_score = self.clip_crop_text_score(
            image_path=image_path,
            bbox=decision.bbox_xyxy,
            text=clip_text,
        )

        final_score = self.compute_final_score(decision, clip_score=clip_score)

        checks = []
        failed_reasons = []

        if not PIL_AVAILABLE:
            checks.append(False)
            failed_reasons.append("pil_not_available_for_bbox_validation")
        elif clipped_bbox is None:
            checks.append(False)
            failed_reasons.append("invalid_bbox_after_clipping")
        else:
            area_ok = self.min_area_ratio <= area_ratio <= self.max_area_ratio
            checks.append(area_ok)
            if not area_ok:
                failed_reasons.append(
                    f"bad_area_ratio={area_ratio:.5f}, allowed=[{self.min_area_ratio}, {self.max_area_ratio}]"
                )

        det_ok = float(decision.det_score) >= self.min_det_score
        checks.append(det_ok)
        if not det_ok:
            failed_reasons.append(f"low_det_score={decision.det_score:.3f}")

        relation_ok = float(decision.relation_score) >= self.min_relation_score
        checks.append(relation_ok)
        if not relation_ok:
            failed_reasons.append(f"low_relation_score={decision.relation_score:.3f}")

        context_ok = float(decision.context_score) >= self.min_context_score
        checks.append(context_ok)
        if not context_ok:
            failed_reasons.append(f"low_context_score={decision.context_score:.3f}")

        final_ok = final_score >= self.accept_threshold
        checks.append(final_ok)
        if not final_ok:
            failed_reasons.append(f"low_final_score={final_score:.3f}")

        accepted = all(checks)

        if accepted:
            reason = "accepted"
        else:
            reason = "rejected_" + "|".join(failed_reasons)

        return VerificationResult(
            target_object=decision.target_object,
            source_object=decision.source_object,
            selected_uid=decision.selected_uid,
            accepted=bool(accepted),
            final_score=float(final_score),
            total_score=float(decision.total_score),
            det_score=float(decision.det_score),
            relation_score=float(decision.relation_score),
            context_score=float(decision.context_score),
            visual_score=float(decision.visual_score),
            descriptor_score=float(decision.descriptor_score),
            base_det_score=float(decision.base_det_score),
            clip_score=clip_score,
            area_ratio=float(area_ratio),
            bbox_xyxy_clipped=clipped_bbox,
            reason=reason,
            operation=decision.operation,
            attribute=decision.attribute,
            value=decision.value,
            anchor_object=decision.anchor_object,
            source_object_text=decision.source_object_text,
            target_object_text=decision.target_object_text,
            anchor_object_text=decision.anchor_object_text,
            source_descriptors=decision.source_descriptors,
            target_descriptors=decision.target_descriptors,
            anchor_descriptors=decision.anchor_descriptors,
            matched_relation=decision.matched_relation,
            matched_partner_uid=decision.matched_partner_uid,
            matched_partner_object=decision.matched_partner_object,
            anchor_uid=decision.anchor_uid,
            anchor_bbox_xyxy=decision.anchor_bbox_xyxy,
            placement_bbox_xyxy=decision.placement_bbox_xyxy,
            placement_relation=decision.placement_relation,
        )

    # ========================================================
    # Visualization helpers
    # ========================================================
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

    def visualize_verification(
        self,
        image_path: str,
        graph: RelationGraph,
        result_key: str,
        verification: VerificationResult,
        out_path: str,
    ):
        if not PIL_AVAILABLE:
            txt_path = os.path.splitext(out_path)[0] + ".txt"

            lines = [
                "Target verification visualization text fallback",
                "",
                f"result_key: {result_key}",
                f"operation: {verification.operation}",
                f"target: {verification.target_object}",
                f"source: {verification.source_object}",
                f"source_text: {verification.source_object_text}",
                f"selected_uid: {verification.selected_uid}",
                f"accepted: {verification.accepted}",
                f"final_score: {verification.final_score:.3f}",
                f"total_score: {verification.total_score:.3f}",
                f"det_score: {verification.det_score:.3f}",
                f"visual_score: {verification.visual_score:.3f}",
                f"relation_score: {verification.relation_score:.3f}",
                f"context_score: {verification.context_score:.3f}",
                f"area_ratio: {verification.area_ratio:.5f}",
                f"reason: {verification.reason}",
                "",
                "candidates:",
            ]

            for node in graph.nodes.values():
                lines.append(
                    f"- uid={node.uid} | {node.object_name} | score={node.score:.2f} | "
                    f"caption={node.caption} | bbox={node.bbox_xyxy}"
                )

            safe_ensure_parent(txt_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"[Target Validator] PIL is not installed. Saved text visualization: {txt_path}")
            return

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = get_font(15)
        small_font = get_font(13)

        img_w, img_h = image.size
        selected_uid = verification.selected_uid

        bottom_lines = [
            "Target verification",
            "green box = selected target and accepted",
            "red box = selected target but rejected",
            "yellow box = other candidates",
            "",
            f"result_key: {result_key}",
            f"operation: {verification.operation}",
            f"target: {verification.target_object}",
            f"source: {verification.source_object}",
            f"source_text: {verification.source_object_text}",
            f"selected_uid: {verification.selected_uid}",
            f"accepted: {verification.accepted}",
            f"final_score: {verification.final_score:.3f}",
            f"total_score: {verification.total_score:.3f}",
            (
                f"det: {verification.det_score:.3f}, "
                f"visual: {verification.visual_score:.3f}, "
                f"rel: {verification.relation_score:.3f}, "
                f"ctx: {verification.context_score:.3f}"
            ),
            f"area_ratio: {verification.area_ratio:.5f}",
            f"reason: {verification.reason}",
            "",
            "candidates:",
        ]

        for node in graph.nodes.values():
            if not node.bbox_xyxy or len(node.bbox_xyxy) != 4:
                bottom_lines.append(
                    f"- uid={node.uid} | {node.object_name} | invalid bbox={node.bbox_xyxy}"
                )
                continue

            x1, y1, x2, y2 = node.bbox_xyxy

            x1, x2 = sorted([float(x1), float(x2)])
            y1, y2 = sorted([float(y1), float(y2)])

            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                bottom_lines.append(
                    f"- uid={node.uid} | {node.object_name} | invalid bbox={node.bbox_xyxy}"
                )
                continue

            if node.uid == selected_uid:
                color = "green" if verification.accepted else "red"
                width = 5
                role = "selected"
            else:
                color = "yellow"
                width = 2
                role = "candidate"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            inner_label_raw = f"{node.uid}:{node.object_name}"
            inner_label = self._fit_text_to_width(
                draw=draw,
                text=inner_label_raw,
                font=small_font,
                max_width=max(20, int(x2 - x1 - 8)),
            )

            if inner_label:
                tw, th = self._text_size(draw, inner_label, small_font)

                tag_x1 = x1
                tag_y1 = y1
                tag_x2 = min(img_w - 1, x1 + tw + 8)
                tag_y2 = min(img_h - 1, y1 + th + 6)

                if tag_x2 > tag_x1 and tag_y2 > tag_y1:
                    draw.rectangle(
                        [tag_x1, tag_y1, tag_x2, tag_y2],
                        fill="black",
                        outline=color,
                    )
                    draw.text(
                        (tag_x1 + 4, tag_y1 + 3),
                        inner_label,
                        fill="white",
                        font=small_font,
                    )

            bottom_lines.append(
                f"- uid={node.uid} | {node.object_name} | score={node.score:.2f} | "
                f"base={node.base_det_score:.2f} | desc={node.descriptor_score:.2f} | "
                f"role={role} | color={color} | bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) | "
                f"text={node.object_text}"
            )

        image = self._append_bottom_text_panel(
            image=image,
            title="Target verification",
            lines=bottom_lines,
            font=font,
            padding=10,
            line_gap=5,
            panel_fill="black",
            text_fill="white",
        )

        safe_ensure_parent(out_path)
        image.save(out_path)


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
# Run validator
# ============================================================
def run_validator(
    task: ParsedTask,
    sample_dir: str,
    parsed_task_path: str,
    candidates_json: Optional[str] = None,
    decisions_json: Optional[str] = None,
    accept_threshold: float = 0.45,
    min_det_score: float = 0.15,
    min_relation_score: float = 0.0,
    min_context_score: float = 0.0,
    min_area_ratio: float = 0.0002,
    max_area_ratio: float = 0.98,
    enable_clip: bool = False,
    clip_path: Optional[str] = None,
    device: Optional[str] = None,
):
    candidates_json = candidates_json or os.path.join(sample_dir, "01_dino_candidates.json")
    decisions_json = decisions_json or os.path.join(sample_dir, "04_context_decisions.json")

    if not os.path.exists(candidates_json):
        raise FileNotFoundError(
            f"Cannot find DINO candidates: {candidates_json}\n"
            "Please run dino_detector.py first with the same input."
        )

    if not os.path.exists(decisions_json):
        raise FileNotFoundError(
            f"Cannot find context decisions: {decisions_json}\n"
            "Please run context_checker.py first with the same input."
        )

    candidates = load_candidates(candidates_json)
    decisions = load_decisions(decisions_json)
    graph = RelationGraph.from_candidates(candidates)

    validator = TargetValidator(
        accept_threshold=accept_threshold,
        min_det_score=min_det_score,
        min_relation_score=min_relation_score,
        min_context_score=min_context_score,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        enable_clip=enable_clip,
        clip_model_name_or_path=clip_path,
        device=device,
    )

    results: Dict[str, VerificationResult] = {}

    for decision_key, decision in decisions.items():
        result = validator.verify(
            image_path=task.image_path,
            decision=decision,
        )

        results[decision_key] = result

        vis_name = f"05_validation_{safe_filename(decision_key)}.jpg"

        validator.visualize_verification(
            image_path=task.image_path,
            graph=graph,
            result_key=decision_key,
            verification=result,
            out_path=os.path.join(sample_dir, vis_name),
        )

    results_path = os.path.join(sample_dir, "05_verification_results.json")
    save_verification_results(results, results_path)

    print("=" * 80)
    print("[Target Validator] Using parsed task + context decisions")
    print(f"[Target Validator] parsed task:      {parsed_task_path}")
    print(f"[Target Validator] candidates_json:  {candidates_json}")
    print(f"[Target Validator] decisions_json:   {decisions_json}")
    print(f"[Target Validator] sample_id:        {task.sample_id}")
    print(f"[Target Validator] image_path:       {task.image_path}")
    print(f"[Target Validator] sample_dir:       {sample_dir}")

    print("[Target Validator] results:")
    for key, result in results.items():
        print(
            f"  {key}: selected={result.selected_uid}, accepted={result.accepted}, "
            f"final={result.final_score:.3f}, area={result.area_ratio:.5f}, "
            f"reason={result.reason}"
        )

    print(f"[Target Validator] saved results: {results_path}")

    if PIL_AVAILABLE:
        print("[Target Validator] saved visualizations: 05_validation_*.jpg")
    else:
        print("[Target Validator] saved visualization text files: 05_validation_*.txt")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run target validation from previous StructEdit outputs. "
            "This script reads 00_parsed_task.json, 01_dino_candidates.json, "
            "and 04_context_decisions.json."
        )
    )

    # Mode 1: original PLE annotation, but loads existing parsed task.
    parser.add_argument("--input-json", type=str, default="", help="Original PLE annotation JSON path.")
    parser.add_argument("--idx", type=int, default=0, help="Sample index for --input-json.")

    # Mode 2: parsed task from command parser.
    parser.add_argument("--parsed-json", type=str, default="", help="Parsed task JSON path, e.g. outputs/command/63/00_parsed_task.json.")

    # Optional explicit dependency paths.
    parser.add_argument("--candidates-json", type=str, default="", help="Optional explicit path to 01_dino_candidates.json.")
    parser.add_argument("--decisions-json", type=str, default="", help="Optional explicit path to 04_context_decisions.json.")

    # Validation thresholds.
    parser.add_argument("--accept-threshold", type=float, default=0.45)
    parser.add_argument("--min-det-score", type=float, default=0.15)
    parser.add_argument("--min-relation-score", type=float, default=0.0)
    parser.add_argument("--min-context-score", type=float, default=0.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0002)
    parser.add_argument("--max-area-ratio", type=float, default=0.98)

    # Optional CLIP check.
    parser.add_argument("--enable-clip", action="store_true")
    parser.add_argument("--clip-path", type=str, default="")
    parser.add_argument("--device", type=str, default="", help="Optional device, e.g. cuda or cpu.")

    args = parser.parse_args()

    task, sample_dir, parsed_task_path = build_task_and_sample_dir(args)

    run_validator(
        task=task,
        sample_dir=sample_dir,
        parsed_task_path=parsed_task_path,
        candidates_json=args.candidates_json or None,
        decisions_json=args.decisions_json or None,
        accept_threshold=args.accept_threshold,
        min_det_score=args.min_det_score,
        min_relation_score=args.min_relation_score,
        min_context_score=args.min_context_score,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        enable_clip=args.enable_clip,
        clip_path=args.clip_path or None,
        device=args.device or None,
    )


if __name__ == "__main__":
    main()
