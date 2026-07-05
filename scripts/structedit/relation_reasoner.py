import os
import re
import sys
import json
import math
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple

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
class GraphEdge:
    src_id: str
    src_name: str
    dst_id: str
    dst_name: str
    predicate: str
    score: float

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
# Candidate / edge loaders
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


def load_edges(path: str) -> List[GraphEdge]:
    data = load_json(path)

    return [
        GraphEdge(
            src_id=x.get("src_id", ""),
            src_name=clean_phrase(x.get("src_name", "")),
            dst_id=x.get("dst_id", ""),
            dst_name=clean_phrase(x.get("dst_name", "")),
            predicate=clean_phrase(x.get("predicate", "")),
            score=float(x.get("score", 0.0)),
        )
        for x in data
    ]


def save_decisions(decisions: Dict[str, CandidateDecision], path: str):
    save_json({k: v.to_dict() for k, v in decisions.items()}, path)


# ============================================================
# Geometry helpers
# ============================================================
def bbox_area(box: List[float]) -> float:
    if not box or len(box) != 4:
        return 0.0

    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter + 1e-6

    return inter / union


def center_xy(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def spatial_score_from_bbox(
    box: List[float],
    value: str,
    image_width: int,
    image_height: int,
) -> float:
    if not box or len(box) != 4:
        return 0.5

    cx, cy = center_xy(box)
    nx = cx / max(1.0, float(image_width))
    ny = cy / max(1.0, float(image_height))

    value = clean_phrase(value)

    if value in ["left", "leftmost"]:
        return float(max(0.0, min(1.0, 1.0 - nx)))

    if value in ["right", "rightmost"]:
        return float(max(0.0, min(1.0, nx)))

    if value in ["top", "upper", "topmost"]:
        return float(max(0.0, min(1.0, 1.0 - ny)))

    if value in ["bottom", "lower", "bottommost"]:
        return float(max(0.0, min(1.0, ny)))

    if value in ["center", "middle", "central"]:
        dx = nx - 0.5
        dy = ny - 0.5
        dist = math.sqrt(dx * dx + dy * dy)
        return float(max(0.0, 1.0 - dist / 0.7071))

    return 0.5


def descriptor_position_score(
    node: GraphNode,
    descriptors: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> float:
    scores = []

    for d in descriptors or []:
        dtype = clean_phrase(d.get("type", ""))
        value = clean_phrase(d.get("value", ""))

        if dtype in ["position", "spatial"] and value:
            scores.append(
                spatial_score_from_bbox(
                    node.bbox_xyxy,
                    value,
                    image_width,
                    image_height,
                )
            )

    if not scores:
        return 0.5

    return float(sum(scores) / len(scores))


def duplicate_candidate_quality(node: GraphNode) -> float:
    desc_bonus = 0.08 if node.descriptors else 0.0
    text_bonus = min(0.04, len(clean_phrase(node.object_text).split()) * 0.01)
    return float(node.score + desc_bonus + text_bonus)


def selected_bbox_conflict(
    box: List[float],
    selected_boxes: List[List[float]],
    threshold: float,
) -> bool:
    for b in selected_boxes:
        if bbox_iou(box, b) >= threshold:
            return True
    return False


# ============================================================
# Relation graph
# ============================================================
class RelationGraph:
    def __init__(self, nodes: Dict[str, GraphNode]):
        self.nodes = nodes

    @classmethod
    def from_candidates(
        cls,
        candidates: List[DetectionCandidate],
        nms_iou: float = 0.92,
    ):
        raw_nodes: List[GraphNode] = []

        for cand in candidates:
            if not cand.uid:
                continue

            raw_nodes.append(
                GraphNode(
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
            )

        # NMS per object name to suppress duplicate boxes generated by:
        #   object with descriptors + same plain relation object
        kept: List[GraphNode] = []

        for node in sorted(raw_nodes, key=duplicate_candidate_quality, reverse=True):
            same_object_kept = [
                k for k in kept
                if clean_phrase(k.object_name) == clean_phrase(node.object_name)
                or clean_phrase(k.query_object) == clean_phrase(node.query_object)
            ]

            duplicate = False
            for k in same_object_kept:
                if bbox_iou(node.bbox_xyxy, k.bbox_xyxy) >= nms_iou:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(node)

        nodes = {n.uid: n for n in kept}
        return cls(nodes)

    def nodes_by_object(self, object_name: str) -> List[GraphNode]:
        object_name = clean_phrase(object_name)

        return [
            n for n in self.nodes.values()
            if clean_phrase(n.object_name) == object_name
            or clean_phrase(n.query_object) == object_name
        ]


# ============================================================
# Reasoner
# ============================================================
class RelationConsistencyReasoner:
    """
    Fourth step:
      00 parsed task
      01 DINO candidates
      02 relation edges
      -> 03 selected edit instances
    """

    def __init__(
        self,
        det_weight: float = 0.50,
        relation_weight: float = 0.50,
        avoid_same_box: bool = True,
        selected_iou_threshold: float = 0.85,
        nms_iou: float = 0.92,
        image_width: int = 1,
        image_height: int = 1,
    ):
        self.det_weight = float(det_weight)
        self.relation_weight = float(relation_weight)
        self.avoid_same_box = avoid_same_box
        self.selected_iou_threshold = float(selected_iou_threshold)
        self.nms_iou = float(nms_iou)
        self.image_width = int(image_width)
        self.image_height = int(image_height)

    def get_unit_attr(self, unit: Any, key: str, default=None):
        if isinstance(unit, dict):
            return unit.get(key, default)
        return getattr(unit, key, default)

    def build_edge_lookup(self, edges: List[GraphEdge], graph: RelationGraph) -> Dict[tuple, float]:
        lookup = {}

        valid = set(graph.nodes.keys())

        for e in edges:
            if e.src_id not in valid or e.dst_id not in valid:
                continue

            lookup[(e.src_id, e.dst_id, clean_phrase(e.predicate))] = float(e.score)

        return lookup

    def compute_visual_score(self, node: GraphNode) -> float:
        """
        Use descriptor-aware DINO score directly.

        dino_detector.py already combines:
        base_det_score + descriptor_score

        For commands like "top knife", node.score is already corrected.
        Do not recompute it here using raw base_det_score.
        """
        return float(getattr(node, "score", getattr(node, "base_det_score", 0.0)))

    def get_owner_specs(self, unit: EditUnit) -> List[Dict[str, Any]]:
        """
        Extract owner relation from descriptors.

        Example:
          source_object = shirt
          source_descriptors = [
            {
              "type": "owner",
              "value": "person",
              "relation": "wearing",
              "owner_descriptors": [{"type":"position","value":"left"}]
            }
          ]

        This means:
          person --wearing--> shirt
          and owner person should satisfy left.
        """
        specs = []

        source_object = clean_phrase(self.get_unit_attr(unit, "source_object", ""))

        for d in self.get_unit_attr(unit, "source_descriptors", []) or []:
            if clean_phrase(d.get("type", "")) != "owner":
                continue

            owner = clean_phrase(d.get("value", ""))
            predicate = clean_phrase(d.get("relation", "")) or "with"
            owner_desc = d.get("owner_descriptors", []) or []

            if owner and source_object:
                specs.append(
                    {
                        "owner": owner,
                        "predicate": predicate,
                        "object": source_object,
                        "owner_descriptors": owner_desc,
                    }
                )

        return specs

    def score_owner_relation_support(
        self,
        node: GraphNode,
        unit: EditUnit,
        graph: RelationGraph,
        edge_lookup: Dict[tuple, float],
    ) -> Tuple[Optional[float], str, str, str]:
        """
        Unit-specific owner scoring.

        For:
          change the left person's shirt to blue

        Candidate node = shirt.
        We score:
          left person --wearing--> this shirt
        """
        node_obj = clean_phrase(node.object_name)
        specs = self.get_owner_specs(unit)

        if not specs:
            return None, "", "", ""

        best = None
        best_reason = ""
        best_partner_uid = ""
        best_partner_obj = ""

        for spec in specs:
            owner = spec["owner"]
            predicate = spec["predicate"]
            obj = spec["object"]
            owner_descriptors = spec["owner_descriptors"]

            if node_obj != obj and clean_phrase(node.query_object) != obj:
                continue

            owner_nodes = graph.nodes_by_object(owner)

            for owner_node in owner_nodes:
                edge_score = edge_lookup.get(
                    (owner_node.uid, node.uid, predicate),
                    None,
                )

                if edge_score is None:
                    continue

                owner_desc_score = descriptor_position_score(
                    owner_node,
                    owner_descriptors,
                    self.image_width,
                    self.image_height,
                )

                # Use both relation edge and descriptor on the owner.
                # For "left person's shirt", this strongly prefers the left person.
                combined = float(0.70 * edge_score + 0.30 * owner_desc_score)

                if best is None or combined > best:
                    best = combined
                    best_reason = (
                        f"owner_relation_support: "
                        f"{owner_node.uid}({owner}) --{predicate}:{edge_score:.3f}--> "
                        f"{node.uid}({obj}), owner_desc={owner_desc_score:.3f}"
                    )
                    best_partner_uid = owner_node.uid
                    best_partner_obj = owner

        return best, best_reason, best_partner_uid, best_partner_obj

    def score_general_relation_support(
        self,
        node: GraphNode,
        graph: RelationGraph,
        relations: List[RelationSpec],
        edge_lookup: Dict[tuple, float],
    ) -> Tuple[float, str, str, str]:
        """
        General relation support.

        Examples:
          cat --wearing--> hat
          person --wearing--> shirt
        """
        node_obj = clean_phrase(node.object_name)
        node_query = clean_phrase(node.query_object)

        relevant_seen = False
        best = None
        best_reason = ""
        best_partner_uid = ""
        best_partner_obj = ""

        for rel in relations:
            subject = clean_phrase(rel.subject)
            predicate = clean_phrase(rel.predicate)
            obj = clean_phrase(rel.object)

            if not subject or not predicate or not obj:
                continue

            # node is subject
            if node_obj == subject or node_query == subject:
                relevant_seen = True
                other_nodes = graph.nodes_by_object(obj)

                for other in other_nodes:
                    score = edge_lookup.get((node.uid, other.uid, predicate), None)
                    if score is None:
                        continue

                    if best is None or score > best:
                        best = float(score)
                        best_reason = (
                            f"general_relation_support: "
                            f"{node.uid}({subject}) --{predicate}:{score:.3f}--> "
                            f"{other.uid}({obj})"
                        )
                        best_partner_uid = other.uid
                        best_partner_obj = obj

            # node is object
            if node_obj == obj or node_query == obj:
                relevant_seen = True
                other_nodes = graph.nodes_by_object(subject)

                for other in other_nodes:
                    score = edge_lookup.get((other.uid, node.uid, predicate), None)
                    if score is None:
                        continue

                    if best is None or score > best:
                        best = float(score)
                        best_reason = (
                            f"general_relation_support: "
                            f"{other.uid}({subject}) --{predicate}:{score:.3f}--> "
                            f"{node.uid}({obj})"
                        )
                        best_partner_uid = other.uid
                        best_partner_obj = subject

        if best is not None:
            return best, best_reason, best_partner_uid, best_partner_obj

        if relevant_seen:
            return 0.0, "relation_relevant_but_no_edge_found", "", ""

        return 0.5, "no_relevant_relation_for_this_object", "", ""

    def unit_decision_key(self, idx: int, unit: EditUnit) -> str:
        obj = (
            self.get_unit_attr(unit, "target_object_text", "")
            or self.get_unit_attr(unit, "target_object", "")
            or self.get_unit_attr(unit, "source_object_text", "")
            or self.get_unit_attr(unit, "source_object", "")
            or self.get_unit_attr(unit, "anchor_object_text", "")
            or self.get_unit_attr(unit, "anchor_object", "")
            or f"unit_{idx}"
        )

        return f"{idx}_{clean_phrase(obj).replace(' ', '_')}"

    def relation_score_for_unit_candidate(
        self,
        node: GraphNode,
        unit: EditUnit,
        graph: RelationGraph,
        relations: List[RelationSpec],
        edge_lookup: Dict[tuple, float],
    ) -> Tuple[float, str, str, str]:
        owner_score, owner_reason, owner_partner_uid, owner_partner_obj = self.score_owner_relation_support(
            node=node,
            unit=unit,
            graph=graph,
            edge_lookup=edge_lookup,
        )

        op = clean_phrase(self.get_unit_attr(unit, "operation", ""))
        edit_type = int(self.get_unit_attr(unit, "edit_type", -1))
        anchor_object = clean_phrase(self.get_unit_attr(unit, "anchor_object", ""))

        # For ADD/MOVE, the parsed spatial relation describes the requested
        # destination. It usually does not hold in the source image, so using
        # it as source-instance evidence selects the wrong object.
        if (op in ["add", "move"] or edit_type in [0, 3]) and anchor_object:
            if owner_score is not None:
                return owner_score, owner_reason, owner_partner_uid, owner_partner_obj
            return 0.5, "target_spatial_relation_reserved_for_placement", "", ""

        general_score, general_reason, general_partner_uid, general_partner_obj = self.score_general_relation_support(
            node=node,
            graph=graph,
            relations=relations,
            edge_lookup=edge_lookup,
        )

        if owner_score is not None:
            # Owner relation is more specific to commands like:
            # "left person's shirt"
            if owner_score >= general_score:
                return owner_score, owner_reason, owner_partner_uid, owner_partner_obj

        return general_score, general_reason, general_partner_uid, general_partner_obj

    def select_anchor_node(
        self,
        unit: EditUnit,
        graph: RelationGraph,
        exclude_uid: Optional[str] = None,
    ) -> Optional[GraphNode]:
        anchor_object = clean_phrase(self.get_unit_attr(unit, "anchor_object", ""))
        if not anchor_object:
            return None

        candidates = graph.nodes_by_object(anchor_object)
        candidates = [n for n in candidates if not exclude_uid or n.uid != exclude_uid]
        if not candidates:
            return None

        anchor_text = clean_phrase(self.get_unit_attr(unit, "anchor_object_text", anchor_object))
        anchor_desc = self.get_unit_attr(unit, "anchor_descriptors", []) or []

        def anchor_score(node: GraphNode) -> float:
            score = self.compute_visual_score(node)
            if anchor_desc:
                score = 0.80 * score + 0.20 * descriptor_position_score(
                    node, anchor_desc, self.image_width, self.image_height
                )
            if anchor_text and clean_phrase(node.object_text) == anchor_text:
                score += 0.03
            return float(score)

        return max(candidates, key=anchor_score)

    def compute_placement_bbox(
        self,
        anchor_bbox: Optional[List[float]],
        relation: str,
        source_bbox: Optional[List[float]] = None,
    ) -> Optional[List[float]]:
        if not anchor_bbox or len(anchor_bbox) != 4:
            return None

        ax1, ay1, ax2, ay2 = [float(v) for v in anchor_bbox]
        aw = max(4.0, ax2 - ax1)
        ah = max(4.0, ay2 - ay1)
        acx = 0.5 * (ax1 + ax2)
        acy = 0.5 * (ay1 + ay2)

        if source_bbox and len(source_bbox) == 4:
            sx1, sy1, sx2, sy2 = [float(v) for v in source_bbox]
            obj_w = max(4.0, sx2 - sx1)
            obj_h = max(4.0, sy2 - sy1)
        else:
            # ADD has no source object. Anchor-sized is a neutral first estimate;
            # downstream generation can resize the synthesized object inside it.
            obj_w = aw
            obj_h = ah

        obj_w = min(obj_w, max(4.0, float(self.image_width)))
        obj_h = min(obj_h, max(4.0, float(self.image_height)))
        gap = max(6.0, 0.12 * max(aw, ah))
        relation = clean_phrase(relation)

        if relation == "left_of":
            x2 = ax1 - gap
            x1 = x2 - obj_w
            y1 = acy - obj_h / 2.0
        elif relation == "right_of":
            x1 = ax2 + gap
            x2 = x1 + obj_w
            y1 = acy - obj_h / 2.0
        elif relation == "above":
            y2 = ay1 - gap
            y1 = y2 - obj_h
            x1 = acx - obj_w / 2.0
        elif relation == "below":
            y1 = ay2 + gap
            y2 = y1 + obj_h
            x1 = acx - obj_w / 2.0
        elif relation in ["inside", "in", "on"]:
            scale = 0.80 if relation in ["inside", "in"] else 1.0
            obj_w = min(obj_w, aw * scale)
            obj_h = min(obj_h, ah * scale)
            x1 = acx - obj_w / 2.0
            y1 = acy - obj_h / 2.0
        elif relation in ["near", "beside"]:
            right_space = float(self.image_width) - ax2
            left_space = ax1
            if right_space >= left_space:
                x1 = ax2 + gap
                x2 = x1 + obj_w
            else:
                x2 = ax1 - gap
                x1 = x2 - obj_w
            y1 = acy - obj_h / 2.0
        else:
            # Depth relations cannot be represented exactly in 2-D. Keep a
            # slightly offset, overlapping box as a usable generation region.
            x1 = acx - obj_w / 2.0 + 0.15 * aw
            y1 = acy - obj_h / 2.0 + 0.10 * ah

        if relation not in ["left_of", "right_of", "near", "beside"]:
            x2 = x1 + obj_w
        if relation not in ["above", "below"]:
            y2 = y1 + obj_h

        max_x1 = max(0.0, float(self.image_width) - obj_w)
        max_y1 = max(0.0, float(self.image_height) - obj_h)
        x1 = max(0.0, min(max_x1, x1))
        y1 = max(0.0, min(max_y1, y1))
        x2 = x1 + obj_w
        y2 = y1 + obj_h

        return [float(x1), float(y1), float(x2), float(y2)]

    def attach_placement_plan(
        self,
        decision: CandidateDecision,
        unit: EditUnit,
        graph: RelationGraph,
    ) -> CandidateDecision:
        op = clean_phrase(self.get_unit_attr(unit, "operation", ""))
        edit_type = int(self.get_unit_attr(unit, "edit_type", -1))
        anchor_object = clean_phrase(self.get_unit_attr(unit, "anchor_object", ""))
        relation = clean_phrase(self.get_unit_attr(unit, "relation", ""))

        if not anchor_object or not (op in ["add", "move"] or edit_type in [0, 3]):
            return decision

        anchor_node = self.select_anchor_node(
            unit=unit,
            graph=graph,
            exclude_uid=decision.selected_uid,
        )
        if anchor_node is None:
            decision.reason += "; placement_anchor_not_found"
            decision.placement_relation = relation
            return decision

        decision.anchor_uid = anchor_node.uid
        decision.anchor_bbox_xyxy = [float(v) for v in anchor_node.bbox_xyxy]
        decision.placement_relation = relation
        decision.placement_bbox_xyxy = self.compute_placement_bbox(
            anchor_bbox=decision.anchor_bbox_xyxy,
            relation=relation,
            source_bbox=decision.bbox_xyxy if op == "move" or edit_type == 3 else None,
        )
        decision.matched_partner_uid = anchor_node.uid
        decision.matched_partner_object = anchor_object
        decision.reason += "; placement_anchor_localized"
        return decision

    def rank_candidates_for_unit(
        self,
        unit_index: int,
        unit: EditUnit,
        graph: RelationGraph,
        relations: List[RelationSpec],
        edge_lookup: Dict[tuple, float],
    ) -> List[CandidateDecision]:
        target_object = clean_phrase(self.get_unit_attr(unit, "target_object", ""))
        source_object = clean_phrase(self.get_unit_attr(unit, "source_object", ""))

        # Add operation does not select existing source object.
        if not source_object:
            return []

        candidates = graph.nodes_by_object(source_object)
        ranked = []

        for node in candidates:
            visual_score = self.compute_visual_score(node)
            base_det_score = float(getattr(node, "base_det_score", node.score))
            descriptor_score = float(getattr(node, "descriptor_score", 0.5))
            descriptor_scores = getattr(node, "descriptor_scores", {}) or {}

            relation_score, relation_reason, partner_uid, partner_obj = self.relation_score_for_unit_candidate(
                node=node,
                unit=unit,
                graph=graph,
                relations=relations,
                edge_lookup=edge_lookup,
            )

            total_score = (
                self.det_weight * float(visual_score)
                + self.relation_weight * float(relation_score)
            )

            # Small bonus when the candidate was produced from the exact described object text.
            unit_text = clean_phrase(self.get_unit_attr(unit, "source_object_text", ""))
            node_text = clean_phrase(node.object_text)

            if unit_text and node_text == unit_text:
                total_score += 0.03
                relation_reason += "; exact_source_object_text_match_bonus=0.03"

            ranked.append(
                CandidateDecision(
                    unit_index=unit_index,
                    target_object=target_object,
                    source_object=source_object,
                    selected_uid=node.uid,
                    bbox_xyxy=node.bbox_xyxy,
                    det_score=float(visual_score),
                    relation_score=float(relation_score),
                    context_score=0.5,
                    total_score=float(total_score),
                    reason=(
                        "selected_by_visual_score_and_relation_score; "
                        + relation_reason
                    ),
                    operation=self.get_unit_attr(unit, "operation", ""),
                    attribute=self.get_unit_attr(unit, "attribute", ""),
                    value=self.get_unit_attr(unit, "value", ""),
                    anchor_object=self.get_unit_attr(unit, "anchor_object", ""),
                    source_object_text=self.get_unit_attr(unit, "source_object_text", source_object),
                    target_object_text=self.get_unit_attr(unit, "target_object_text", target_object),
                    anchor_object_text=self.get_unit_attr(
                        unit,
                        "anchor_object_text",
                        self.get_unit_attr(unit, "anchor_object", ""),
                    ),
                    source_descriptors=self.get_unit_attr(unit, "source_descriptors", []) or [],
                    target_descriptors=self.get_unit_attr(unit, "target_descriptors", []) or [],
                    anchor_descriptors=self.get_unit_attr(unit, "anchor_descriptors", []) or [],
                    descriptor_score=descriptor_score,
                    descriptor_scores=descriptor_scores,
                    visual_score=float(visual_score),
                    base_det_score=base_det_score,
                    matched_relation=relation_reason,
                    matched_partner_uid=partner_uid,
                    matched_partner_object=partner_obj,
                )
            )

        ranked.sort(key=lambda x: x.total_score, reverse=True)
        return ranked

    def select_targets(
        self,
        edit_units: List[EditUnit],
        graph: RelationGraph,
        relations: List[RelationSpec],
        edges: List[GraphEdge],
    ) -> Dict[str, CandidateDecision]:
        edge_lookup = self.build_edge_lookup(edges, graph)

        decisions: Dict[str, CandidateDecision] = {}
        used_uids = set()
        used_boxes: List[List[float]] = []

        for idx, unit in enumerate(edit_units):
            op = clean_phrase(self.get_unit_attr(unit, "operation", ""))
            edit_type = int(self.get_unit_attr(unit, "edit_type", -1))

            # ADD has no source instance. Localize only the anchor and derive
            # a destination box from the requested spatial relation.
            if op == "add" or edit_type == 0:
                key = self.unit_decision_key(idx, unit)
                anchor_node = self.select_anchor_node(unit=unit, graph=graph)
                relation = clean_phrase(self.get_unit_attr(unit, "relation", ""))

                if anchor_node is None:
                    decisions[key] = CandidateDecision(
                        unit_index=idx,
                        target_object=clean_phrase(self.get_unit_attr(unit, "target_object", "")),
                        source_object="",
                        selected_uid=None,
                        bbox_xyxy=None,
                        det_score=0.0,
                        relation_score=0.0,
                        context_score=0.0,
                        total_score=0.0,
                        reason="add_operation_anchor_not_found",
                        operation=op,
                        attribute=self.get_unit_attr(unit, "attribute", ""),
                        value=self.get_unit_attr(unit, "value", ""),
                        anchor_object=self.get_unit_attr(unit, "anchor_object", ""),
                        source_object_text="",
                        target_object_text=self.get_unit_attr(unit, "target_object_text", ""),
                        anchor_object_text=self.get_unit_attr(unit, "anchor_object_text", ""),
                        source_descriptors=self.get_unit_attr(unit, "source_descriptors", []) or [],
                        target_descriptors=self.get_unit_attr(unit, "target_descriptors", []) or [],
                        anchor_descriptors=self.get_unit_attr(unit, "anchor_descriptors", []) or [],
                        placement_relation=relation,
                    )
                else:
                    anchor_score = self.compute_visual_score(anchor_node)
                    placement_box = self.compute_placement_bbox(
                        anchor_bbox=anchor_node.bbox_xyxy,
                        relation=relation,
                        source_bbox=None,
                    )
                    decisions[key] = CandidateDecision(
                        unit_index=idx,
                        target_object=clean_phrase(self.get_unit_attr(unit, "target_object", "")),
                        source_object="",
                        selected_uid=None,
                        bbox_xyxy=None,
                        det_score=float(anchor_score),
                        relation_score=1.0 if placement_box is not None else 0.0,
                        context_score=0.5,
                        total_score=float(anchor_score),
                        reason="add_operation_anchor_localized_and_placement_box_created",
                        operation=op,
                        attribute=self.get_unit_attr(unit, "attribute", ""),
                        value=self.get_unit_attr(unit, "value", ""),
                        anchor_object=self.get_unit_attr(unit, "anchor_object", ""),
                        source_object_text="",
                        target_object_text=self.get_unit_attr(unit, "target_object_text", ""),
                        anchor_object_text=self.get_unit_attr(unit, "anchor_object_text", ""),
                        source_descriptors=self.get_unit_attr(unit, "source_descriptors", []) or [],
                        target_descriptors=self.get_unit_attr(unit, "target_descriptors", []) or [],
                        anchor_descriptors=self.get_unit_attr(unit, "anchor_descriptors", []) or [],
                        visual_score=float(anchor_score),
                        base_det_score=float(getattr(anchor_node, "base_det_score", anchor_score)),
                        descriptor_score=float(getattr(anchor_node, "descriptor_score", 0.5)),
                        descriptor_scores=getattr(anchor_node, "descriptor_scores", {}) or {},
                        matched_partner_uid=anchor_node.uid,
                        matched_partner_object=clean_phrase(self.get_unit_attr(unit, "anchor_object", "")),
                        anchor_uid=anchor_node.uid,
                        anchor_bbox_xyxy=[float(v) for v in anchor_node.bbox_xyxy],
                        placement_bbox_xyxy=placement_box,
                        placement_relation=relation,
                    )
                continue

            ranked = self.rank_candidates_for_unit(
                unit_index=idx,
                unit=unit,
                graph=graph,
                relations=relations,
                edge_lookup=edge_lookup,
            )

            selected = None

            for cand in ranked:
                if self.avoid_same_box:
                    if cand.selected_uid in used_uids:
                        continue

                    if cand.bbox_xyxy is not None and selected_bbox_conflict(
                        cand.bbox_xyxy,
                        used_boxes,
                        self.selected_iou_threshold,
                    ):
                        continue

                selected = cand
                break

            if selected is None and ranked:
                selected = ranked[0]

            key = self.unit_decision_key(idx, unit)

            target_object = clean_phrase(unit.target_object)
            source_object = clean_phrase(unit.source_object)

            if selected is None:
                if unit.operation == "add":
                    reason = "add_operation_has_no_existing_source_object_to_select"
                elif not source_object:
                    reason = "no_source_object_for_this_edit_unit"
                else:
                    reason = f"no_candidate_found_for_{source_object}"

                decisions[key] = CandidateDecision(
                    unit_index=idx,
                    target_object=target_object,
                    source_object=source_object,
                    selected_uid=None,
                    bbox_xyxy=None,
                    det_score=0.0,
                    relation_score=0.0,
                    context_score=0.0,
                    total_score=0.0,
                    reason=reason,
                    operation=self.get_unit_attr(unit, "operation", ""),
                    attribute=self.get_unit_attr(unit, "attribute", ""),
                    value=self.get_unit_attr(unit, "value", ""),
                    anchor_object=self.get_unit_attr(unit, "anchor_object", ""),
                    source_object_text=self.get_unit_attr(unit, "source_object_text", source_object),
                    target_object_text=self.get_unit_attr(unit, "target_object_text", target_object),
                    anchor_object_text=self.get_unit_attr(
                        unit,
                        "anchor_object_text",
                        self.get_unit_attr(unit, "anchor_object", ""),
                    ),
                    source_descriptors=self.get_unit_attr(unit, "source_descriptors", []) or [],
                    target_descriptors=self.get_unit_attr(unit, "target_descriptors", []) or [],
                    anchor_descriptors=self.get_unit_attr(unit, "anchor_descriptors", []) or [],
                    descriptor_score=0.0,
                    descriptor_scores={},
                    visual_score=0.0,
                    base_det_score=0.0,
                )
            else:
                selected = self.attach_placement_plan(selected, unit, graph)
                decisions[key] = selected

                if selected.selected_uid is not None:
                    used_uids.add(selected.selected_uid)

                if selected.bbox_xyxy is not None:
                    used_boxes.append(selected.bbox_xyxy)

        return decisions

    # ========================================================
    # Visualization
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

    def visualize_decisions(
        self,
        image_path: str,
        graph: RelationGraph,
        decisions: Dict[str, CandidateDecision],
        out_path: str,
    ):
        if not PIL_AVAILABLE:
            txt_path = os.path.splitext(out_path)[0] + ".txt"

            lines = [
                "Relation consistency reasoning visualization text fallback",
                "",
                "decisions:",
            ]

            for key, d in decisions.items():
                lines.append(
                    f"- {key}: op={d.operation}, target={d.target_object}, "
                    f"source={d.source_object}, uid={d.selected_uid}, "
                    f"det={d.det_score:.3f}, rel={d.relation_score:.3f}, "
                    f"visual={d.visual_score:.3f}, total={d.total_score:.3f}, "
                    f"reason={d.reason}"
                )

            lines.extend(["", "candidates:"])
            for node in graph.nodes.values():
                lines.append(
                    f"- uid={node.uid} | {node.object_name} | score={node.score:.2f} | "
                    f"caption={node.caption} | bbox={node.bbox_xyxy}"
                )

            safe_ensure_parent(txt_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"[Relation Reasoner] PIL is not installed. Saved text visualization: {txt_path}")
            return

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = get_font(15)
        small_font = get_font(13)

        img_w, img_h = image.size

        selected_uids = {
            d.selected_uid
            for d in decisions.values()
            if d.selected_uid is not None
        }

        uid_to_key = {
            d.selected_uid: key
            for key, d in decisions.items()
            if d.selected_uid is not None
        }

        bottom_lines = [
            "Relation consistency reasoning: final target selection",
            "red box = finally selected edit instance",
            "yellow box = unselected candidate after NMS",
            "",
            "decisions:",
        ]

        for key, d in decisions.items():
            bottom_lines.append(
                f"- {key}: op={d.operation}, source={d.source_object}, target={d.target_object}, "
                f"uid={d.selected_uid}, det={d.det_score:.3f}, visual={d.visual_score:.3f}, "
                f"rel={d.relation_score:.3f}, total={d.total_score:.3f}, "
                f"partner={d.matched_partner_uid}, reason={d.reason}"
            )

        bottom_lines.extend(["", "candidates after NMS:"])

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

            is_selected = node.uid in selected_uids

            color = "red" if is_selected else "yellow"
            width = 5 if is_selected else 2

            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            if is_selected:
                key = uid_to_key.get(node.uid, "unknown_unit")
                inner_label_raw = f"{node.uid}:{key}"
                role = "selected"
            else:
                inner_label_raw = f"{node.uid}:{node.object_name}"
                role = "candidate"

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
                f"role={role} | bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) | "
                f"text={node.object_text}"
            )

        image = self._append_bottom_text_panel(
            image=image,
            title="Relation consistency reasoning",
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


def get_image_size(image_path: str) -> Tuple[int, int]:
    if not PIL_AVAILABLE:
        return 1, 1

    if not image_path or not os.path.exists(image_path):
        return 1, 1

    img = Image.open(image_path).convert("RGB")
    return img.size


# ============================================================
# Run reasoner
# ============================================================
def run_reasoner(
    task: ParsedTask,
    sample_dir: str,
    parsed_task_path: str,
    candidates_json: Optional[str] = None,
    edges_json: Optional[str] = None,
    det_weight: float = 0.50,
    relation_weight: float = 0.50,
    allow_same_box: bool = False,
    nms_iou: float = 0.92,
    selected_iou_threshold: float = 0.85,
):
    candidates_json = candidates_json or os.path.join(sample_dir, "01_dino_candidates.json")
    edges_json = edges_json or os.path.join(sample_dir, "02_relation_edges.json")

    if not os.path.exists(candidates_json):
        raise FileNotFoundError(
            f"Cannot find DINO candidates: {candidates_json}\n"
            "Please run dino_detector.py first with the same input."
        )

    if not os.path.exists(edges_json):
        raise FileNotFoundError(
            f"Cannot find relation edges: {edges_json}\n"
            "Please run relation_graph.py first with the same input."
        )

    candidates = load_candidates(candidates_json)
    edges = load_edges(edges_json)

    graph = RelationGraph.from_candidates(candidates, nms_iou=nms_iou)

    image_width, image_height = get_image_size(task.image_path)

    reasoner = RelationConsistencyReasoner(
        det_weight=det_weight,
        relation_weight=relation_weight,
        avoid_same_box=not allow_same_box,
        selected_iou_threshold=selected_iou_threshold,
        nms_iou=nms_iou,
        image_width=image_width,
        image_height=image_height,
    )

    decisions = reasoner.select_targets(
        edit_units=task.edit_units,
        graph=graph,
        relations=task.relations,
        edges=edges,
    )

    decisions_path = os.path.join(sample_dir, "03_relation_decisions.json")
    vis_path = os.path.join(sample_dir, "03_relation_reasoning_selected.jpg")

    save_decisions(decisions, decisions_path)

    reasoner.visualize_decisions(
        image_path=task.image_path,
        graph=graph,
        decisions=decisions,
        out_path=vis_path,
    )

    print("=" * 80)
    print("[Relation Reasoner] Using parsed task + DINO candidates + relation edges")
    print(f"[Relation Reasoner] parsed task:      {parsed_task_path}")
    print(f"[Relation Reasoner] candidates_json:  {candidates_json}")
    print(f"[Relation Reasoner] edges_json:       {edges_json}")
    print(f"[Relation Reasoner] sample_id:        {task.sample_id}")
    print(f"[Relation Reasoner] image_path:       {task.image_path}")
    print(f"[Relation Reasoner] sample_dir:       {sample_dir}")
    print(f"[Relation Reasoner] graph nodes:      {len(graph.nodes)}")
    print("[Relation Reasoner] final decisions:")

    for key, d in decisions.items():
        print(
            f"  {key}: op={d.operation}, source={d.source_object}, target={d.target_object}, "
            f"selected={d.selected_uid}, det={d.det_score:.3f}, visual={d.visual_score:.3f}, "
            f"rel={d.relation_score:.3f}, total={d.total_score:.3f}, "
            f"partner={d.matched_partner_uid}, reason={d.reason}"
        )

    print(f"[Relation Reasoner] saved decisions: {decisions_path}")

    if PIL_AVAILABLE:
        print(f"[Relation Reasoner] saved vis:       {vis_path}")
    else:
        print(f"[Relation Reasoner] saved vis text:  {os.path.splitext(vis_path)[0] + '.txt'}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run relation consistency reasoning from previous StructEdit outputs. "
            "This script reads 00_parsed_task.json, 01_dino_candidates.json, "
            "and 02_relation_edges.json."
        )
    )

    # Mode 1: original PLE annotation, but loads existing parsed task.
    parser.add_argument("--input-json", type=str, default="", help="Original PLE annotation JSON path.")
    parser.add_argument("--idx", type=int, default=0, help="Sample index for --input-json.")

    # Mode 2: parsed task from command parser.
    parser.add_argument("--parsed-json", type=str, default="", help="Parsed task JSON path, e.g. outputs/command/1/00_parsed_task.json.")

    # Optional explicit dependency paths.
    parser.add_argument("--candidates-json", type=str, default="", help="Optional explicit path to 01_dino_candidates.json.")
    parser.add_argument("--edges-json", type=str, default="", help="Optional explicit path to 02_relation_edges.json.")

    # Reasoning parameters.
    parser.add_argument("--det-weight", type=float, default=0.50)
    parser.add_argument("--relation-weight", type=float, default=0.50)
    parser.add_argument("--allow-same-box", action="store_true")
    parser.add_argument("--nms-iou", type=float, default=0.92)
    parser.add_argument("--selected-iou-threshold", type=float, default=0.85)

    args = parser.parse_args()

    task, sample_dir, parsed_task_path = build_task_and_sample_dir(args)

    run_reasoner(
        task=task,
        sample_dir=sample_dir,
        parsed_task_path=parsed_task_path,
        candidates_json=args.candidates_json or None,
        edges_json=args.edges_json or None,
        det_weight=args.det_weight,
        relation_weight=args.relation_weight,
        allow_same_box=args.allow_same_box,
        nms_iou=args.nms_iou,
        selected_iou_threshold=args.selected_iou_threshold,
    )


if __name__ == "__main__":
    main()
