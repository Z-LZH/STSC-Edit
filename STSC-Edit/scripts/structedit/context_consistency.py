import os
import re
import sys
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

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


@dataclass
class ContextResult:
    target_object: str
    source_object: str
    selected_uid: Optional[str]

    context_score: float
    support_count: int
    support_edges: List[Dict[str, Any]]
    reason: str

    operation: str = ""
    attribute: str = ""
    value: str = ""
    anchor_object: str = ""

    matched_partner_uid: str = ""
    matched_partner_object: str = ""

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
# Candidate / edge / decision loaders
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


def save_decisions(decisions: Dict[str, CandidateDecision], path: str):
    save_json({k: v.to_dict() for k, v in decisions.items()}, path)


def save_context_results(results: Dict[str, ContextResult], path: str):
    save_json({k: v.to_dict() for k, v in results.items()}, path)


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
# Context consistency checker
# ============================================================
class ContextConsistencyChecker:
    """
    Fifth step:
      00 parsed task
      01 DINO candidates
      02 relation edges
      03 relation decisions
      -> 04 context consistency results

    This version supports both:
      - selected-to-selected context, e.g. cat decision + hat decision
      - selected-to-context-node support, e.g. shirt decision supported by person node
    """

    def __init__(self, context_weight: float = 0.25):
        self.context_weight = float(context_weight)

    def build_edge_lookup(self, edges: List[GraphEdge]) -> Dict[Tuple[str, str, str], float]:
        lookup = {}

        for e in edges:
            lookup[(e.src_id, e.dst_id, clean_phrase(e.predicate))] = float(e.score)

        return lookup

    def build_source_to_decisions(
        self,
        decisions: Dict[str, CandidateDecision],
    ) -> Dict[str, List[CandidateDecision]]:
        mapping: Dict[str, List[CandidateDecision]] = {}

        for _, decision in decisions.items():
            source = clean_phrase(decision.source_object)

            if not source:
                continue

            mapping.setdefault(source, []).append(decision)

        return mapping

    def get_pair_edge_score(
        self,
        src_uid: str,
        dst_uid: str,
        predicate: str,
        edge_lookup: Dict[Tuple[str, str, str], float],
    ) -> Optional[float]:
        key = (src_uid, dst_uid, clean_phrase(predicate))

        if key in edge_lookup:
            return edge_lookup[key]

        return None

    def find_best_graph_support(
        self,
        current_uid: str,
        current_source: str,
        relations: List[RelationSpec],
        graph: RelationGraph,
        edge_lookup: Dict[Tuple[str, str, str], float],
    ) -> List[Dict[str, Any]]:
        """
        Find support from graph nodes even if the neighbor object is not an edit decision.

        Example:
          edit decision selects shirt.
          relation is person --wearing--> shirt.
          There may be no person decision, but person nodes exist in graph.
        """
        supports: List[Dict[str, Any]] = []
        current_source = clean_phrase(current_source)

        for rel in relations:
            subject = clean_phrase(rel.subject)
            predicate = clean_phrase(rel.predicate)
            obj = clean_phrase(rel.object)

            if not subject or not predicate or not obj:
                continue

            # Current selected node is relation subject.
            if current_source == subject:
                for other in graph.nodes_by_object(obj):
                    score = self.get_pair_edge_score(
                        src_uid=current_uid,
                        dst_uid=other.uid,
                        predicate=predicate,
                        edge_lookup=edge_lookup,
                    )

                    if score is None:
                        continue

                    supports.append(
                        {
                            "src_uid": current_uid,
                            "src_object": current_source,
                            "dst_uid": other.uid,
                            "dst_object": obj,
                            "predicate": predicate,
                            "score": float(score),
                            "support_type": "graph_context_node",
                        }
                    )

            # Current selected node is relation object.
            if current_source == obj:
                for other in graph.nodes_by_object(subject):
                    score = self.get_pair_edge_score(
                        src_uid=other.uid,
                        dst_uid=current_uid,
                        predicate=predicate,
                        edge_lookup=edge_lookup,
                    )

                    if score is None:
                        continue

                    supports.append(
                        {
                            "src_uid": other.uid,
                            "src_object": subject,
                            "dst_uid": current_uid,
                            "dst_object": current_source,
                            "predicate": predicate,
                            "score": float(score),
                            "support_type": "graph_context_node",
                        }
                    )

        supports.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return supports

    def find_selected_decision_support(
        self,
        decision: CandidateDecision,
        decisions: Dict[str, CandidateDecision],
        relations: List[RelationSpec],
        edge_lookup: Dict[Tuple[str, str, str], float],
    ) -> List[Dict[str, Any]]:
        supports: List[Dict[str, Any]] = []

        if decision.selected_uid is None:
            return supports

        source_to_decisions = self.build_source_to_decisions(decisions)

        current_source = clean_phrase(decision.source_object)
        current_uid = decision.selected_uid

        for rel in relations:
            subject = clean_phrase(rel.subject)
            predicate = clean_phrase(rel.predicate)
            obj = clean_phrase(rel.object)

            if not subject or not predicate or not obj:
                continue

            # Current selected node is relation subject.
            if current_source == subject:
                other_decisions = source_to_decisions.get(obj, [])

                for other_decision in other_decisions:
                    if other_decision.selected_uid is None:
                        continue

                    score = self.get_pair_edge_score(
                        src_uid=current_uid,
                        dst_uid=other_decision.selected_uid,
                        predicate=predicate,
                        edge_lookup=edge_lookup,
                    )

                    if score is None:
                        continue

                    supports.append(
                        {
                            "src_uid": current_uid,
                            "src_object": current_source,
                            "dst_uid": other_decision.selected_uid,
                            "dst_object": obj,
                            "predicate": predicate,
                            "score": float(score),
                            "support_type": "selected_decision_pair",
                        }
                    )

            # Current selected node is relation object.
            if current_source == obj:
                other_decisions = source_to_decisions.get(subject, [])

                for other_decision in other_decisions:
                    if other_decision.selected_uid is None:
                        continue

                    score = self.get_pair_edge_score(
                        src_uid=other_decision.selected_uid,
                        dst_uid=current_uid,
                        predicate=predicate,
                        edge_lookup=edge_lookup,
                    )

                    if score is None:
                        continue

                    supports.append(
                        {
                            "src_uid": other_decision.selected_uid,
                            "src_object": subject,
                            "dst_uid": current_uid,
                            "dst_object": current_source,
                            "predicate": predicate,
                            "score": float(score),
                            "support_type": "selected_decision_pair",
                        }
                    )

        supports.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return supports

    def relation_is_relevant(self, source_object: str, relations: List[RelationSpec]) -> bool:
        source_object = clean_phrase(source_object)

        for rel in relations:
            if source_object in [clean_phrase(rel.subject), clean_phrase(rel.object)]:
                return True

        return False

    def deduplicate_support_edges(self, supports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []

        for s in supports:
            key = (
                s.get("src_uid", ""),
                s.get("dst_uid", ""),
                s.get("predicate", ""),
            )

            if key in seen:
                continue

            seen.add(key)
            out.append(s)

        return out

    def check_one_decision(
        self,
        decision: CandidateDecision,
        decisions: Dict[str, CandidateDecision],
        relations: List[RelationSpec],
        graph: RelationGraph,
        edge_lookup: Dict[Tuple[str, str, str], float],
    ) -> ContextResult:

        op = clean_phrase(decision.operation)
        if op == "add":
            ok = decision.anchor_uid is not None and decision.placement_bbox_xyxy is not None
            return ContextResult(
                target_object=decision.target_object,
                source_object=decision.source_object,
                selected_uid=None,
                context_score=1.0 if ok else 0.0,
                support_count=1 if ok else 0,
                support_edges=[],
                reason=(
                    "add_anchor_localized_and_destination_planned"
                    if ok else "add_anchor_or_destination_missing"
                ),
                operation=decision.operation,
                attribute=decision.attribute,
                value=decision.value,
                anchor_object=decision.anchor_object,
                matched_partner_uid=decision.anchor_uid or "",
                matched_partner_object=decision.anchor_object,
            )

        if op == "move" and decision.anchor_object:
            ok = (
                decision.selected_uid is not None
                and decision.anchor_uid is not None
                and decision.placement_bbox_xyxy is not None
            )
            return ContextResult(
                target_object=decision.target_object,
                source_object=decision.source_object,
                selected_uid=decision.selected_uid,
                context_score=1.0 if ok else 0.0,
                support_count=1 if ok else 0,
                support_edges=[],
                reason=(
                    "move_source_and_anchor_localized_destination_planned"
                    if ok else "move_source_anchor_or_destination_missing"
                ),
                operation=decision.operation,
                attribute=decision.attribute,
                value=decision.value,
                anchor_object=decision.anchor_object,
                matched_partner_uid=decision.anchor_uid or "",
                matched_partner_object=decision.anchor_object,
            )

        if decision.selected_uid is None:
            return ContextResult(
                target_object=decision.target_object,
                source_object=decision.source_object,
                selected_uid=None,
                context_score=0.0,
                support_count=0,
                support_edges=[],
                reason="no_selected_node",
                operation=decision.operation,
                attribute=decision.attribute,
                value=decision.value,
                anchor_object=decision.anchor_object,
            )

        current_source = clean_phrase(decision.source_object)
        current_uid = decision.selected_uid

        selected_supports = self.find_selected_decision_support(
            decision=decision,
            decisions=decisions,
            relations=relations,
            edge_lookup=edge_lookup,
        )

        graph_supports = self.find_best_graph_support(
            current_uid=current_uid,
            current_source=current_source,
            relations=relations,
            graph=graph,
            edge_lookup=edge_lookup,
        )

        all_supports = selected_supports + graph_supports
        all_supports = self.deduplicate_support_edges(all_supports)

        if decision.matched_partner_uid:
            # Prefer visual trace from relation_reasoner if available.
            for s in all_supports:
                if decision.matched_partner_uid in [s.get("src_uid"), s.get("dst_uid")]:
                    s["support_type"] = s.get("support_type", "") + "+matched_partner_from_reasoner"

        if not all_supports:
            if self.relation_is_relevant(current_source, relations):
                return ContextResult(
                    target_object=decision.target_object,
                    source_object=decision.source_object,
                    selected_uid=decision.selected_uid,
                    context_score=0.0,
                    support_count=0,
                    support_edges=[],
                    reason="relation_relevant_but_no_support_edge_found",
                    operation=decision.operation,
                    attribute=decision.attribute,
                    value=decision.value,
                    anchor_object=decision.anchor_object,
                    matched_partner_uid=decision.matched_partner_uid,
                    matched_partner_object=decision.matched_partner_object,
                )

            return ContextResult(
                target_object=decision.target_object,
                source_object=decision.source_object,
                selected_uid=decision.selected_uid,
                context_score=0.5,
                support_count=0,
                support_edges=[],
                reason="no_relevant_context_relation_for_this_decision",
                operation=decision.operation,
                attribute=decision.attribute,
                value=decision.value,
                anchor_object=decision.anchor_object,
                matched_partner_uid=decision.matched_partner_uid,
                matched_partner_object=decision.matched_partner_object,
            )

        # Use best support rather than mean, because one strong structural edge is enough.
        scores = [float(s.get("score", 0.0)) for s in all_supports]
        context_score = float(max(scores))

        return ContextResult(
            target_object=decision.target_object,
            source_object=decision.source_object,
            selected_uid=decision.selected_uid,
            context_score=context_score,
            support_count=len(all_supports),
            support_edges=all_supports,
            reason="context_supported_by_relation_graph",
            operation=decision.operation,
            attribute=decision.attribute,
            value=decision.value,
            anchor_object=decision.anchor_object,
            matched_partner_uid=decision.matched_partner_uid,
            matched_partner_object=decision.matched_partner_object,
        )

    def update_decision(
        self,
        decision: CandidateDecision,
        context_result: ContextResult,
    ) -> CandidateDecision:
        old_total = float(decision.total_score)
        context_score = float(context_result.context_score)

        new_total = (
            (1.0 - self.context_weight) * old_total
            + self.context_weight * context_score
        )

        decision.context_score = context_score
        decision.total_score = float(new_total)

        if "+context_consistency" not in decision.reason:
            decision.reason += "+context_consistency"

        return decision

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

    def visualize_context(
        self,
        image_path: str,
        graph: RelationGraph,
        decision_key: str,
        decision: CandidateDecision,
        context_result: ContextResult,
        out_path: str,
    ):
        if not PIL_AVAILABLE:
            txt_path = os.path.splitext(out_path)[0] + ".txt"

            lines = [
                "Context consistency visualization text fallback",
                "",
                f"decision_key: {decision_key}",
                f"operation: {decision.operation}",
                f"target: {decision.target_object}",
                f"source: {decision.source_object}",
                f"selected_uid: {decision.selected_uid}",
                f"context_score: {context_result.context_score:.3f}",
                f"support_count: {context_result.support_count}",
                f"updated_total: {decision.total_score:.3f}",
                f"reason: {context_result.reason}",
                "",
                "support_edges:",
            ]

            if context_result.support_edges:
                for e in context_result.support_edges:
                    lines.append(
                        f"- {e['src_uid']}({e['src_object']}) "
                        f"--{e['predicate']}:{e['score']:.3f}--> "
                        f"{e['dst_uid']}({e['dst_object']}) "
                        f"[{e.get('support_type', '')}]"
                    )
            else:
                lines.append("- none")

            lines.extend(["", "nodes:"])
            for node in graph.nodes.values():
                lines.append(
                    f"- uid={node.uid} | {node.object_name} | score={node.score:.2f} | "
                    f"caption={node.caption} | bbox={node.bbox_xyxy}"
                )

            safe_ensure_parent(txt_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"[Context Consistency] PIL is not installed. Saved text visualization: {txt_path}")
            return

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = get_font(15)
        small_font = get_font(13)

        img_w, img_h = image.size
        current_uid = decision.selected_uid

        support_neighbor_uids = set()
        support_pairs = []

        for e in context_result.support_edges:
            src_uid = e["src_uid"]
            dst_uid = e["dst_uid"]
            predicate = e["predicate"]
            score = e["score"]
            support_type = e.get("support_type", "")

            support_pairs.append((src_uid, dst_uid, predicate, score, support_type))

            if src_uid != current_uid:
                support_neighbor_uids.add(src_uid)

            if dst_uid != current_uid:
                support_neighbor_uids.add(dst_uid)

        bottom_lines = [
            "Context consistency",
            "red box = current selected edit target",
            "green box = context-supporting neighbor",
            "yellow box = other DINO candidate",
            "cyan line = structural context support",
            "",
            f"decision_key: {decision_key}",
            f"operation: {decision.operation}",
            f"target: {decision.target_object}",
            f"source: {decision.source_object}",
            f"selected_uid: {decision.selected_uid}",
            f"context_score: {context_result.context_score:.3f}",
            f"support_count: {context_result.support_count}",
            f"updated_total: {decision.total_score:.3f}",
            f"reason: {context_result.reason}",
            "",
            "nodes:",
        ]

        uid_to_clipped_box = {}

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

            uid_to_clipped_box[node.uid] = (x1, y1, x2, y2)

            if node.uid == current_uid:
                color = "red"
                width = 5
                role = "current_target"
            elif node.uid in support_neighbor_uids:
                color = "green"
                width = 5
                role = "support_neighbor"
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

        bottom_lines.extend(["", "support edges:"])

        if not support_pairs:
            bottom_lines.append("- none")

        for src_uid, dst_uid, predicate, score, support_type in support_pairs:
            if src_uid not in uid_to_clipped_box or dst_uid not in uid_to_clipped_box:
                bottom_lines.append(
                    f"- {src_uid} --{predicate}:{score:.3f}--> {dst_uid} | "
                    "skipped because bbox is missing or invalid"
                )
                continue

            src_box = uid_to_clipped_box[src_uid]
            dst_box = uid_to_clipped_box[dst_uid]

            sc = (
                (src_box[0] + src_box[2]) / 2.0,
                (src_box[1] + src_box[3]) / 2.0,
            )
            dc = (
                (dst_box[0] + dst_box[2]) / 2.0,
                (dst_box[1] + dst_box[3]) / 2.0,
            )

            draw.line(
                [float(sc[0]), float(sc[1]), float(dc[0]), float(dc[1])],
                fill="cyan",
                width=3,
            )

            mx = float((sc[0] + dc[0]) / 2.0)
            my = float((sc[1] + dc[1]) / 2.0)

            score_label = f"{score:.2f}"
            tw, th = self._text_size(draw, score_label, small_font)

            label_x1 = max(0, min(mx, img_w - tw - 8))
            label_y1 = max(0, min(my, img_h - th - 6))
            label_x2 = min(img_w - 1, label_x1 + tw + 8)
            label_y2 = min(img_h - 1, label_y1 + th + 6)

            if label_x2 > label_x1 and label_y2 > label_y1:
                draw.rectangle(
                    [label_x1, label_y1, label_x2, label_y2],
                    fill="black",
                    outline="cyan",
                )
                draw.text(
                    (label_x1 + 4, label_y1 + 3),
                    score_label,
                    fill="white",
                    font=small_font,
                )

            bottom_lines.append(
                f"- {src_uid} --{predicate}:{score:.3f}--> {dst_uid} [{support_type}]"
            )

        image = self._append_bottom_text_panel(
            image=image,
            title="Context consistency",
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
# Run context checker
# ============================================================
def run_context_checker(
    task: ParsedTask,
    sample_dir: str,
    parsed_task_path: str,
    candidates_json: Optional[str] = None,
    edges_json: Optional[str] = None,
    decisions_json: Optional[str] = None,
    context_weight: float = 0.25,
):
    candidates_json = candidates_json or os.path.join(sample_dir, "01_dino_candidates.json")
    edges_json = edges_json or os.path.join(sample_dir, "02_relation_edges.json")
    decisions_json = decisions_json or os.path.join(sample_dir, "03_relation_decisions.json")

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

    if not os.path.exists(decisions_json):
        raise FileNotFoundError(
            f"Cannot find relation decisions: {decisions_json}\n"
            "Please run relation_reasoner.py first with the same input."
        )

    candidates = load_candidates(candidates_json)
    edges = load_edges(edges_json)
    decisions = load_decisions(decisions_json)

    graph = RelationGraph.from_candidates(candidates)

    checker = ContextConsistencyChecker(context_weight=context_weight)
    edge_lookup = checker.build_edge_lookup(edges)

    context_results: Dict[str, ContextResult] = {}

    for decision_key, decision in list(decisions.items()):
        ctx = checker.check_one_decision(
            decision=decision,
            decisions=decisions,
            relations=task.relations,
            graph=graph,
            edge_lookup=edge_lookup,
        )

        context_results[decision_key] = ctx
        decisions[decision_key] = checker.update_decision(decision, ctx)

        vis_name = f"04_context_{safe_filename(decision_key)}.jpg"

        checker.visualize_context(
            image_path=task.image_path,
            graph=graph,
            decision_key=decision_key,
            decision=decisions[decision_key],
            context_result=ctx,
            out_path=os.path.join(sample_dir, vis_name),
        )

    results_path = os.path.join(sample_dir, "04_context_results.json")
    updated_decisions_path = os.path.join(sample_dir, "04_context_decisions.json")

    save_context_results(context_results, results_path)
    save_decisions(decisions, updated_decisions_path)

    print("=" * 80)
    print("[Context Consistency] Using parsed task + selected decisions + relation edges")
    print(f"[Context Consistency] parsed task:     {parsed_task_path}")
    print(f"[Context Consistency] candidates_json: {candidates_json}")
    print(f"[Context Consistency] edges_json:      {edges_json}")
    print(f"[Context Consistency] decisions_json:  {decisions_json}")
    print(f"[Context Consistency] sample_id:       {task.sample_id}")
    print(f"[Context Consistency] image_path:      {task.image_path}")
    print(f"[Context Consistency] sample_dir:      {sample_dir}")

    print("[Context Consistency] results:")
    for key, ctx in context_results.items():
        print(
            f"  {key}: selected={ctx.selected_uid}, context={ctx.context_score:.3f}, "
            f"support_count={ctx.support_count}, reason={ctx.reason}"
        )

    print(f"[Context Consistency] saved results:   {results_path}")
    print(f"[Context Consistency] saved decisions: {updated_decisions_path}")

    if PIL_AVAILABLE:
        print("[Context Consistency] saved visualizations: 04_context_*.jpg")
    else:
        print("[Context Consistency] saved visualization text files: 04_context_*.txt")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run context consistency checking from previous StructEdit outputs. "
            "This script reads 00_parsed_task.json, 01_dino_candidates.json, "
            "02_relation_edges.json, and 03_relation_decisions.json."
        )
    )

    # Mode 1: original PLE annotation, but loads existing parsed task.
    parser.add_argument("--input-json", type=str, default="", help="Original PLE annotation JSON path.")
    parser.add_argument("--idx", type=int, default=0, help="Sample index for --input-json.")

    # Mode 2: parsed task from command parser.
    parser.add_argument("--parsed-json", type=str, default="", help="Parsed task JSON path, e.g. outputs/command/63/00_parsed_task.json.")

    # Optional explicit dependency paths.
    parser.add_argument("--candidates-json", type=str, default="", help="Optional explicit path to 01_dino_candidates.json.")
    parser.add_argument("--edges-json", type=str, default="", help="Optional explicit path to 02_relation_edges.json.")
    parser.add_argument("--decisions-json", type=str, default="", help="Optional explicit path to 03_relation_decisions.json.")

    # Context parameter.
    parser.add_argument("--context-weight", type=float, default=0.25)

    args = parser.parse_args()

    task, sample_dir, parsed_task_path = build_task_and_sample_dir(args)

    run_context_checker(
        task=task,
        sample_dir=sample_dir,
        parsed_task_path=parsed_task_path,
        candidates_json=args.candidates_json or None,
        edges_json=args.edges_json or None,
        decisions_json=args.decisions_json or None,
        context_weight=args.context_weight,
    )


if __name__ == "__main__":
    main()
