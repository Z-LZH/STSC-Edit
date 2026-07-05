import os
import re
import sys
import json
import math
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple

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


# ============================================================
# Relation patterns for fallback PLE parsing
# ============================================================
RELATION_PATTERNS = [
    ("to the left of", "left_of"),
    ("left of", "left_of"),
    ("to the right of", "right_of"),
    ("right of", "right_of"),
    ("next to", "near"),
    ("near", "near"),
    ("beside", "near"),
    ("close to", "near"),
    ("on top of", "on"),
    ("on", "on"),
    ("above", "above"),
    ("over", "above"),
    ("under", "below"),
    ("below", "below"),
    ("beneath", "below"),
    ("behind", "behind"),
    ("in front of", "in_front_of"),
    ("wearing", "wearing"),
    ("holding", "holding"),
    ("with", "with"),
    ("inside", "inside"),
    ("in", "inside"),
]


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


def get_font(size: int = 14):
    if not PIL_AVAILABLE:
        return None

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ============================================================
# Output path rules
# ============================================================
def infer_ple_output_root(input_json: str) -> str:
    """
    Example:
      data/PLE_bench/0_random_140/export/annotations.json

    Output:
      outputs/PLE_bench/0_random_140
    """
    parts = os.path.normpath(input_json).split(os.sep)

    if "PLE_bench" in parts:
        i = parts.index("PLE_bench")
        if i + 1 < len(parts):
            return os.path.join("outputs", "PLE_bench", parts[i + 1])

    return os.path.join("outputs", "PLE_bench")


def get_ple_sample_dir(input_json: str, sample_id: str) -> str:
    out_root = infer_ple_output_root(input_json)
    return make_sample_out_dir(out_root, sample_id)


def get_command_sample_dir(parsed_json: str, sample_id: str) -> str:
    """
    Normally parsed_json is:
      outputs/command/<id>/00_parsed_task.json

    So sample_dir is dirname(parsed_json).
    """
    parent = os.path.dirname(os.path.abspath(parsed_json))
    if os.path.basename(parent) == str(sample_id):
        return parent

    return make_sample_out_dir(os.path.join("outputs", "command"), sample_id)


# ============================================================
# Parsed task loading
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

    # fallback from edit_units
    if not source_objects:
        source_objects = unique_keep_order([u.source_object for u in edit_units if u.source_object])

    if not target_objects:
        target_objects = unique_keep_order([u.target_object for u in edit_units if u.target_object])

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


def save_parsed_task(task: ParsedTask, path: str):
    save_json(task.to_dict(), path)


# ============================================================
# Fallback PLE parsing
# Only used if outputs/.../<id>/00_parsed_task.json does not exist.
# ============================================================
def extract_bracket_targets(target_prompt: str) -> List[str]:
    targets = re.findall(r"\[([^\]]+)\]", target_prompt or "")
    return [clean_phrase(t) for t in targets if clean_phrase(t)]


def parse_edit_units_from_annotation(record: Dict[str, Any]) -> List[EditUnit]:
    edit_action = safe_json_loads(record.get("edit_action"), {})
    if not isinstance(edit_action, dict):
        edit_action = {}

    units = []

    for target_object, info in edit_action.items():
        if not isinstance(info, dict):
            continue

        target_object = clean_phrase(target_object)
        source_object = clean_phrase(info.get("action", target_object))
        edit_type = int(info.get("edit_type", -1))

        units.append(
            EditUnit(
                target_object=target_object,
                source_object=source_object,
                edit_type=edit_type,
                position=int(info.get("position", -1)),
                operation=EDIT_TYPE_ID_TO_NAME.get(edit_type, ""),
                source_object_text=source_object,
                target_object_text=target_object,
            )
        )

    return units


def find_phrase_span(text: str, phrase: str) -> Optional[Tuple[int, int]]:
    text = normalize_text(text)
    phrase = normalize_text(phrase)

    pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
    m = re.search(pattern, text)
    if m:
        return m.start(), m.end()

    if phrase.endswith("s"):
        short = phrase[:-1]
        pattern = r"\b" + re.escape(short).replace(r"\ ", r"\s+") + r"\b"
        m = re.search(pattern, text)
        if m:
            return m.start(), m.end()

    return None


def infer_relation_between(prompt: str, obj_a: str, obj_b: str) -> Optional[RelationSpec]:
    prompt_norm = normalize_text(prompt)

    span_a = find_phrase_span(prompt_norm, obj_a)
    span_b = find_phrase_span(prompt_norm, obj_b)

    if span_a is None or span_b is None:
        return None

    a_start, a_end = span_a
    b_start, b_end = span_b

    if a_end <= b_start:
        left_obj, right_obj = obj_a, obj_b
        between = prompt_norm[a_end:b_start]
    elif b_end <= a_start:
        left_obj, right_obj = obj_b, obj_a
        between = prompt_norm[b_end:a_start]
    else:
        return RelationSpec(obj_a, "overlap_text", obj_b, "phrase_overlap")

    between = between.strip()

    for pattern, predicate in RELATION_PATTERNS:
        if pattern in between:
            return RelationSpec(
                subject=left_obj,
                predicate=predicate,
                object=right_obj,
                evidence=between,
            )

    if "and" in between:
        return RelationSpec(left_obj, "co_occurs", right_obj, between)

    return None


def extract_relations_from_prompt(prompt: str, objects: List[str]) -> List[RelationSpec]:
    objects = unique_keep_order(objects)
    relations = []
    seen = set()

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            rel = infer_relation_between(prompt, objects[i], objects[j])
            if rel is None:
                continue

            key = (rel.subject, rel.predicate, rel.object)
            if key not in seen:
                seen.add(key)
                relations.append(rel)

    return relations


def parse_annotation_record(record: Dict[str, Any]) -> ParsedTask:
    sample_id = str(record.get("id", "unknown"))
    image_path = record["image"]
    source_prompt = record.get("source_prompt", "")
    target_prompt = record.get("target_prompt", "")

    edit_units = parse_edit_units_from_annotation(record)

    source_objects = unique_keep_order([u.source_object for u in edit_units if u.source_object])
    target_objects = unique_keep_order([u.target_object for u in edit_units if u.target_object])

    for t in extract_bracket_targets(target_prompt):
        if t not in target_objects:
            target_objects.append(t)

    relations = extract_relations_from_prompt(source_prompt, source_objects)

    return ParsedTask(
        sample_id=sample_id,
        image_path=image_path,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        edit_units=edit_units,
        relations=relations,
        source_objects=source_objects,
        target_objects=target_objects,
    )


# ============================================================
# DINO candidates loading
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


# ============================================================
# Geometry helpers
# ============================================================
def get_rel_value(rel: Any, key: str, default: Any = ""):
    if isinstance(rel, dict):
        return rel.get(key, default)
    return getattr(rel, key, default)


def center(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_size(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def area(box):
    w, h = box_size(box)
    return w * h


def iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = area(a) + area(b) - inter + 1e-6

    return inter / union


def x_overlap_ratio(a, b):
    ax1, _, ax2, _ = [float(v) for v in a]
    bx1, _, bx2, _ = [float(v) for v in b]

    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    aw = max(1.0, ax2 - ax1)
    bw = max(1.0, bx2 - bx1)

    return inter / min(aw, bw)


def y_overlap_ratio(a, b):
    _, ay1, _, ay2 = [float(v) for v in a]
    _, by1, _, by2 = [float(v) for v in b]

    inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    ah = max(1.0, ay2 - ay1)
    bh = max(1.0, by2 - by1)

    return inter / min(ah, bh)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


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
                query_object=cand.query_object,
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

    def relation_score(self, src: GraphNode, dst: GraphNode, predicate: str) -> float:
        if not src.bbox_xyxy or not dst.bbox_xyxy:
            return 0.0

        src_box = src.bbox_xyxy
        dst_box = dst.bbox_xyxy

        ac = center(src_box)
        bc = center(dst_box)

        aw, ah = box_size(src_box)
        bw, bh = box_size(dst_box)

        diag = math.sqrt((aw + bw) ** 2 + (ah + bh) ** 2) + 1e-6
        dist = float(np.linalg.norm(ac - bc))
        near_score = max(0.0, 1.0 - dist / (diag * 2.0))

        x_overlap = x_overlap_ratio(src_box, dst_box)

        predicate = clean_phrase(predicate)

        if predicate == "left_of":
            return float(sigmoid((bc[0] - ac[0]) / max(aw, bw)))

        if predicate == "right_of":
            return float(sigmoid((ac[0] - bc[0]) / max(aw, bw)))

        if predicate == "above":
            return float(sigmoid((bc[1] - ac[1]) / max(ah, bh)))

        if predicate == "below":
            return float(sigmoid((ac[1] - bc[1]) / max(ah, bh)))

        if predicate in ["near", "with", "holding"]:
            return float(near_score)

        if predicate == "co_occurs":
            return 0.6

        if predicate == "on":
            vertical_score = sigmoid((bc[1] - ac[1]) / max(ah, bh))
            return float(0.55 * vertical_score + 0.35 * x_overlap + 0.10 * near_score)

        if predicate == "inside":
            x1, y1, x2, y2 = [float(v) for v in src_box]
            X1, Y1, X2, Y2 = [float(v) for v in dst_box]

            strict_inside = float(x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2)
            return max(strict_inside, iou(src_box, dst_box))

        if predicate == "wearing":
            object_above_subject_center = sigmoid((ac[1] - bc[1]) / max(ah, bh))
            score = (
                0.45 * object_above_subject_center
                + 0.35 * x_overlap
                + 0.20 * near_score
            )
            return float(score)

        if predicate in ["behind", "in_front_of"]:
            return float(0.5 * near_score)

        return 0.0

    def build_edges_from_relations(self, relations: List[Any]) -> List[GraphEdge]:
        edges = []

        for rel in relations:
            subject = clean_phrase(get_rel_value(rel, "subject", ""))
            predicate = clean_phrase(get_rel_value(rel, "predicate", ""))
            obj = clean_phrase(get_rel_value(rel, "object", ""))

            if not subject or not predicate or not obj:
                continue

            src_nodes = self.nodes_by_object(subject)
            dst_nodes = self.nodes_by_object(obj)

            for src in src_nodes:
                for dst in dst_nodes:
                    if src.uid == dst.uid:
                        continue

                    score = self.relation_score(src, dst, predicate)

                    edges.append(
                        GraphEdge(
                            src_id=src.uid,
                            src_name=src.object_name,
                            dst_id=dst.uid,
                            dst_name=dst.object_name,
                            predicate=predicate,
                            score=float(score),
                        )
                    )

        edges.sort(key=lambda e: e.score, reverse=True)
        return edges

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

    def visualize_graph(
        self,
        image_path: str,
        relations: List[Any],
        edges: List[GraphEdge],
        out_path: str,
    ):
        if not PIL_AVAILABLE:
            txt_path = os.path.splitext(out_path)[0] + ".txt"
            lines = ["Relation graph visualization text fallback", ""]

            lines.append("relations:")
            if relations:
                for rel in relations:
                    lines.append(
                        f"- {get_rel_value(rel, 'subject')} "
                        f"--{get_rel_value(rel, 'predicate')}--> "
                        f"{get_rel_value(rel, 'object')}"
                    )
            else:
                lines.append("- no explicit parsed relation")

            lines.extend(["", "nodes:"])
            for node in self.nodes.values():
                lines.append(
                    f"- uid={node.uid} | {node.object_name} | score={node.score:.2f} | "
                    f"caption={node.caption} | bbox={node.bbox_xyxy}"
                )

            lines.extend(["", "edges:"])
            if edges:
                for edge in edges:
                    lines.append(
                        f"- {edge.src_id} --{edge.predicate}:{edge.score:.3f}--> {edge.dst_id}"
                    )
            else:
                lines.append("- no relation edges generated")

            safe_ensure_parent(txt_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"[Relation Graph] PIL is not installed. Saved text visualization: {txt_path}")
            return

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = get_font(14)
        small_font = get_font(13)

        img_w, img_h = image.size

        if relations:
            rel_texts = []
            for rel in relations:
                rel_texts.append(
                    f"{get_rel_value(rel, 'subject')} "
                    f"--{get_rel_value(rel, 'predicate')}--> "
                    f"{get_rel_value(rel, 'object')}"
                )
            title = "Relation graph: " + "; ".join(rel_texts)
        else:
            title = "Relation graph: no explicit parsed relation"

        bottom_lines = [
            "yellow box = DINO candidate node",
            "cyan line = relation edge",
            "number on line = relation consistency score",
            "",
            "parsed relations:",
        ]

        if relations:
            for rel in relations:
                bottom_lines.append(
                    f"- {get_rel_value(rel, 'subject')} "
                    f"--{get_rel_value(rel, 'predicate')}--> "
                    f"{get_rel_value(rel, 'object')}"
                )
        else:
            bottom_lines.append("- no explicit parsed relation")

        bottom_lines.extend(["", "nodes:"])

        uid_to_clipped_box = {}

        for node in self.nodes.values():
            if not node.bbox_xyxy or len(node.bbox_xyxy) != 4:
                bottom_lines.append(
                    f"- uid={node.uid} | {node.object_name} | "
                    f"score={node.score:.2f} | invalid bbox={node.bbox_xyxy}"
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
                    f"- uid={node.uid} | {node.object_name} | "
                    f"score={node.score:.2f} | invalid bbox={node.bbox_xyxy}"
                )
                continue

            uid_to_clipped_box[node.uid] = (x1, y1, x2, y2)

            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)

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
                        outline="yellow",
                    )
                    draw.text(
                        (tag_x1 + 4, tag_y1 + 3),
                        inner_label,
                        fill="white",
                        font=small_font,
                    )

            bottom_lines.append(
                f"- uid={node.uid} | {node.object_name} | "
                f"score={node.score:.2f} | base={node.base_det_score:.2f} | "
                f"desc={node.descriptor_score:.2f} | "
                f"bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) | "
                f"text={node.object_text}"
            )

        bottom_lines.extend(["", "edges:"])

        for edge in edges:
            if edge.src_id not in uid_to_clipped_box or edge.dst_id not in uid_to_clipped_box:
                bottom_lines.append(
                    f"- {edge.src_id} --{edge.predicate}:{edge.score:.2f}--> "
                    f"{edge.dst_id} | skipped because node bbox is missing or invalid"
                )
                continue

            src_box = uid_to_clipped_box[edge.src_id]
            dst_box = uid_to_clipped_box[edge.dst_id]

            ac = (
                (src_box[0] + src_box[2]) / 2.0,
                (src_box[1] + src_box[3]) / 2.0,
            )
            bc = (
                (dst_box[0] + dst_box[2]) / 2.0,
                (dst_box[1] + dst_box[3]) / 2.0,
            )

            draw.line(
                [float(ac[0]), float(ac[1]), float(bc[0]), float(bc[1])],
                fill="cyan",
                width=2,
            )

            mx = float((ac[0] + bc[0]) / 2.0)
            my = float((ac[1] + bc[1]) / 2.0)

            score_label = f"{edge.score:.2f}"
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
                f"- {edge.src_id} --{edge.predicate}:{edge.score:.2f}--> {edge.dst_id}"
            )

        if not edges:
            bottom_lines.append("- no relation edges generated")

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


# ============================================================
# Save graph outputs
# ============================================================
def save_edges(edges: List[GraphEdge], path: str):
    save_json([e.to_dict() for e in edges], path)


def save_nodes(graph: RelationGraph, path: str):
    save_json({uid: node.to_dict() for uid, node in graph.nodes.items()}, path)


# ============================================================
# Build task + sample_dir from two allowed input modes
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
    # Mode 1: parsed task from command parser
    # --------------------------------------------------------
    if has_parsed:
        if not os.path.exists(args.parsed_json):
            raise FileNotFoundError(f"Parsed json not found: {args.parsed_json}")

        task = load_parsed_task(args.parsed_json)
        sample_dir = get_command_sample_dir(args.parsed_json, task.sample_id)
        parsed_task_path = args.parsed_json

        return task, sample_dir, parsed_task_path

    # --------------------------------------------------------
    # Mode 2: original PLE annotation
    # --------------------------------------------------------
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    records = load_annotations(args.input_json)

    if args.idx < 0 or args.idx >= len(records):
        raise IndexError(f"--idx {args.idx} out of range, total records={len(records)}")

    record = records[args.idx]
    sample_id = str(record.get("id", "unknown"))
    sample_dir = get_ple_sample_dir(args.input_json, sample_id)

    parsed_task_path = os.path.join(sample_dir, "00_parsed_task.json")

    # Prefer the parsed task generated by step 1 or step 2.
    if os.path.exists(parsed_task_path):
        task = load_parsed_task(parsed_task_path)
    else:
        task = parse_annotation_record(record)
        save_parsed_task(task, parsed_task_path)

    return task, sample_dir, parsed_task_path


# ============================================================
# Main graph runner
# ============================================================
def run_relation_graph(task: ParsedTask, sample_dir: str, parsed_task_path: str):
    candidates_json = os.path.join(sample_dir, "01_dino_candidates.json")

    if not os.path.exists(candidates_json):
        raise FileNotFoundError(
            f"Cannot find DINO candidates: {candidates_json}\n"
            "Please run dino_detector.py first with the same input."
        )

    candidates = load_candidates(candidates_json)

    graph = RelationGraph.from_candidates(candidates)
    edges = graph.build_edges_from_relations(task.relations)

    nodes_path = os.path.join(sample_dir, "02_relation_nodes.json")
    edges_path = os.path.join(sample_dir, "02_relation_edges.json")
    vis_path = os.path.join(sample_dir, "02_relation_graph.jpg")

    save_nodes(graph, nodes_path)
    save_edges(edges, edges_path)

    graph.visualize_graph(
        image_path=task.image_path,
        relations=task.relations,
        edges=edges,
        out_path=vis_path,
    )

    print("=" * 80)
    print("[Relation Graph] Using previous parsed task and DINO candidates")
    print(f"[Relation Graph] parsed task:     {parsed_task_path}")
    print(f"[Relation Graph] candidates json: {candidates_json}")
    print(f"[Relation Graph] sample_id:       {task.sample_id}")
    print(f"[Relation Graph] image_path:      {task.image_path}")
    print(f"[Relation Graph] sample_dir:      {sample_dir}")
    print(f"[Relation Graph] num nodes:       {len(graph.nodes)}")
    print(f"[Relation Graph] num edges:       {len(edges)}")

    if edges:
        print("[Relation Graph] top edges:")
        for e in edges[:10]:
            print(
                f"  {e.src_id}({e.src_name}) "
                f"--{e.predicate}:{e.score:.3f}--> "
                f"{e.dst_id}({e.dst_name})"
            )
    else:
        print("[Relation Graph] no relation edges generated.")

    print(f"[Relation Graph] saved nodes: {nodes_path}")
    print(f"[Relation Graph] saved edges: {edges_path}")

    if PIL_AVAILABLE:
        print(f"[Relation Graph] saved vis:   {vis_path}")
    else:
        print(f"[Relation Graph] saved vis text: {os.path.splitext(vis_path)[0] + '.txt'}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build relation graph from previous StructEdit outputs.\n"
            "This script reads 00_parsed_task.json and 01_dino_candidates.json."
        )
    )

    # Mode 1: original PLE annotation
    parser.add_argument("--input-json", type=str, default="", help="Original PLE annotation JSON path.")
    parser.add_argument("--idx", type=int, default=0, help="Sample index for --input-json.")

    # Mode 2: parsed task from command parser
    parser.add_argument("--parsed-json", type=str, default="", help="Parsed task JSON path, usually outputs/command/<id>/00_parsed_task.json.")

    args = parser.parse_args()

    task, sample_dir, parsed_task_path = build_task_and_sample_dir(args)

    run_relation_graph(
        task=task,
        sample_dir=sample_dir,
        parsed_task_path=parsed_task_path,
    )


if __name__ == "__main__":
    main()
