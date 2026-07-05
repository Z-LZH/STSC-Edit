import os
import re
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from structedit.config import resolve_ann_and_out, get_sample_out_dir, ensure_dir
except Exception:
    def ensure_dir(path: str):
        if path:
            os.makedirs(path, exist_ok=True)

    def get_sample_out_dir(out_dir: str, sample_id: str):
        sample_dir = os.path.join(out_dir, str(sample_id))
        ensure_dir(sample_dir)
        return sample_dir

    def resolve_ann_and_out(ann_rel: str):
        ann_path = ann_rel
        out_dir = "outputs/rule_parse"
        subset_name = os.path.splitext(os.path.basename(ann_rel))[0]
        return ann_path, out_dir, subset_name


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
    "erase": 2,
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


@dataclass
class RelationSpec:
    subject: str
    predicate: str
    object: str
    evidence: str


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
        return asdict(self)


# ============================================================
# Patterns / vocab
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
    ("between", "between"),
    ("wearing", "wearing"),
    ("holding", "holding"),
    ("with", "with"),
    ("inside", "inside"),
    ("in", "inside"),
]

POSITION_PATTERNS = [
    (r"^(?:to\s+)?the\s+left\s+of\s+(?P<anchor>.+)$", "left_of"),
    (r"^(?:to\s+)?left\s+of\s+(?P<anchor>.+)$", "left_of"),
    (r"^(?:to\s+)?the\s+right\s+of\s+(?P<anchor>.+)$", "right_of"),
    (r"^(?:to\s+)?right\s+of\s+(?P<anchor>.+)$", "right_of"),
    (r"^above\s+(?P<anchor>.+)$", "above"),
    (r"^over\s+(?P<anchor>.+)$", "above"),
    (r"^below\s+(?P<anchor>.+)$", "below"),
    (r"^under\s+(?P<anchor>.+)$", "below"),
    (r"^beneath\s+(?P<anchor>.+)$", "below"),
    (r"^behind\s+(?P<anchor>.+)$", "behind"),
    (r"^in\s+front\s+of\s+(?P<anchor>.+)$", "in_front_of"),
    (r"^next\s+to\s+(?P<anchor>.+)$", "near"),
    (r"^near\s+(?P<anchor>.+)$", "near"),
    (r"^beside\s+(?P<anchor>.+)$", "near"),
    (r"^close\s+to\s+(?P<anchor>.+)$", "near"),
    (r"^on\s+top\s+of\s+(?P<anchor>.+)$", "on"),
    (r"^on\s+(?P<anchor>.+)$", "on"),
    (r"^inside\s+(?P<anchor>.+)$", "inside"),
    (r"^in\s+(?P<anchor>.+)$", "inside"),
    (r"^between\s+(?P<anchor>.+)$", "between"),
    (r"^at\s+(?P<anchor>.+)$", "at"),
]

ADD_POSITION_STARTERS = [
    r"to\s+the\s+left\s+of",
    r"to\s+left\s+of",
    r"left\s+of",
    r"to\s+the\s+right\s+of",
    r"to\s+right\s+of",
    r"right\s+of",
    r"in\s+front\s+of",
    r"on\s+top\s+of",
    r"next\s+to",
    r"close\s+to",
    r"beside",
    r"near",
    r"above",
    r"over",
    r"below",
    r"under",
    r"beneath",
    r"behind",
    r"inside",
    r"between",
    r"on",
    r"in",
    r"at",
]

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

ATTRIBUTE_WORDS = [
    "small", "large", "big", "tiny", "tall", "short", "long", "round",
    "old", "new", "dirty", "clean",
]

OWNER_WORDS = {
    "person", "people", "man", "woman", "boy", "girl", "child",
    "male", "female", "cat", "dog", "horse", "bird",
}

CLOTHING_WORDS = {
    "shirt", "t-shirt", "tee", "top", "blouse",
    "coat", "jacket", "sweater", "hoodie",
    "pants", "trousers", "jeans", "shorts", "skirt",
    "dress", "hat", "cap", "helmet",
    "scarf", "tie", "shoe", "shoes", "boot", "boots",
    "glove", "gloves", "sock", "socks",
    "bag", "backpack", "collar",
}

POSITION_DESCRIPTOR_MAP = {
    "left": "left",
    "leftmost": "left",
    "right": "right",
    "rightmost": "right",
    "top": "top",
    "upper": "top",
    "topmost": "top",
    "bottom": "bottom",
    "lower": "bottom",
    "bottommost": "bottom",
    "front": "front",
    "back": "back",
    "middle": "center",
    "center": "center",
    "central": "center",
}


# ============================================================
# Basic utils
# ============================================================
def normalize_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def clean_phrase(x: Any) -> str:
    x = normalize_text(str(x))
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


def load_annotations(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["annotations", "data", "samples"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported annotation json format: {path}")


def save_json(obj, path: str):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def unique_keep_order(xs: List[str]) -> List[str]:
    out = []
    for x in xs:
        x = clean_phrase(x)
        if x and x not in out:
            out.append(x)
    return out


def add_object_once(objects: List[str], obj: str):
    obj = clean_phrase(obj)
    if obj and obj not in ["image", "scene"] and obj not in objects:
        objects.append(obj)


def relation_key(r: RelationSpec):
    return (
        clean_phrase(r.subject),
        clean_phrase(r.predicate),
        clean_phrase(r.object),
    )


def add_relation_once(relations: List[RelationSpec], rel: RelationSpec):
    key = relation_key(rel)
    existing = {relation_key(r) for r in relations}

    if key not in existing:
        relations.append(rel)


# ============================================================
# Object phrase parser
# ============================================================
def _append_descriptor(
    descriptors: List[Dict[str, Any]],
    dtype: str,
    value: str,
    text: str,
    extra: Optional[Dict[str, Any]] = None,
):
    d = {
        "type": dtype,
        "value": clean_phrase(value),
        "text": clean_phrase(text),
    }
    if extra:
        d.update(extra)

    key = json.dumps(d, sort_keys=True, ensure_ascii=False)
    existing = {
        json.dumps(x, sort_keys=True, ensure_ascii=False)
        for x in descriptors
    }
    if key not in existing:
        descriptors.append(d)


def _consume_leading_descriptors(words: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    words = list(words)
    descs: List[Dict[str, Any]] = []

    while len(words) >= 2:
        w = clean_phrase(words[0])

        if w in POSITION_DESCRIPTOR_MAP:
            descs.append({
                "type": "position",
                "value": POSITION_DESCRIPTOR_MAP[w],
                "text": w,
            })
            words = words[1:]
            continue

        if w in COLOR_WORDS:
            descs.append({
                "type": "color",
                "value": w,
                "text": w,
            })
            words = words[1:]
            continue

        if w in MATERIAL_WORDS:
            descs.append({
                "type": "material",
                "value": w,
                "text": w,
            })
            words = words[1:]
            continue

        if w in ATTRIBUTE_WORDS:
            descs.append({
                "type": "attribute",
                "value": w,
                "text": w,
            })
            words = words[1:]
            continue

        break

    return words, descs


def infer_owner_relation(object_name: str) -> str:
    object_name = clean_phrase(object_name)

    if object_name in CLOTHING_WORDS:
        return "wearing"

    return "with"


def parse_described_object(phrase: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Split object phrase into:
      base_object: object sent to DINO
      object_text: original object phrase
      descriptors: modifiers and relation hints

    Examples:
      left knife
        -> base=knife
        -> position:left

      left white dog
        -> base=dog
        -> position:left + color:white

      left person's shirt
        -> base=shirt
        -> owner:person with owner_descriptors=[position:left], relation=wearing

      person wearing pink hat
        -> base=person
        -> relation wearing object=hat with object_descriptors=[color:pink]
    """
    original = clean_phrase(phrase)
    if not original:
        return "", "", []

    descriptors: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Relation phrase:
    # person wearing pink hat
    # dog with blue collar
    # man holding umbrella
    # --------------------------------------------------------
    rel_match = re.match(
        r"^(?P<base>.+?)\s+"
        r"(?P<rel>wearing|holding|with|carrying)\s+"
        r"(?P<obj>.+)$",
        original,
    )

    if rel_match:
        base_part = clean_phrase(rel_match.group("base"))
        rel = clean_phrase(rel_match.group("rel"))
        obj_part = clean_phrase(rel_match.group("obj"))

        base_object, _, base_desc = parse_described_object(base_part)
        obj_base, obj_text, obj_desc = parse_described_object(obj_part)

        descriptors.extend(base_desc)

        descriptors.append({
            "type": "relation",
            "relation": rel,
            "value": f"{rel} {obj_text}",
            "object": obj_base,
            "object_text": obj_text,
            "object_descriptors": obj_desc,
            "text": f"{rel} {obj_text}",
        })

        return base_object, original, descriptors

    words = original.split()

    # Leading descriptors may apply to owner if possessive follows:
    # left person's shirt -> left applies to person
    words_after_leading, leading_descs = _consume_leading_descriptors(words)

    # --------------------------------------------------------
    # Possessive owner:
    # left person's shirt
    # person's blue shirt
    # --------------------------------------------------------
    if len(words_after_leading) >= 2 and words_after_leading[0].endswith("'s"):
        owner = clean_phrase(words_after_leading[0][:-2])
        rest_words = words_after_leading[1:]

        # descriptors after owner apply to base object:
        # person's blue shirt -> blue applies to shirt
        rest_words, base_descs = _consume_leading_descriptors(rest_words)

        base = clean_phrase(" ".join(rest_words)) or original
        relation = infer_owner_relation(base)

        descriptors.extend(base_descs)

        descriptors.append({
            "type": "owner",
            "value": owner,
            "text": words_after_leading[0],
            "relation": relation,
            "owner_descriptors": leading_descs,
        })

        return base, original, descriptors

    # --------------------------------------------------------
    # Non-possessive owner fallback:
    # left person shirt
    # --------------------------------------------------------
    if len(words_after_leading) >= 2 and words_after_leading[0] in OWNER_WORDS:
        owner = clean_phrase(words_after_leading[0])
        rest_words = words_after_leading[1:]

        rest_words, base_descs = _consume_leading_descriptors(rest_words)

        base = clean_phrase(" ".join(rest_words)) or original
        relation = infer_owner_relation(base)

        descriptors.extend(base_descs)

        descriptors.append({
            "type": "owner",
            "value": owner,
            "text": owner,
            "relation": relation,
            "owner_descriptors": leading_descs,
        })

        return base, original, descriptors

    # No owner: leading descriptors apply to object itself
    descriptors.extend(leading_descs)
    base = clean_phrase(" ".join(words_after_leading)) or original

    return base, original, descriptors


# ============================================================
# Relation extraction helpers
# ============================================================
def add_descriptor_relations_for_unit(
    unit: EditUnit,
    relations: List[RelationSpec],
    source_objects: List[str],
):
    """
    Convert descriptors into explicit relations.

    Examples:
      source_object = shirt
      descriptor owner person relation wearing
        -> person --wearing--> shirt

      source_object = person
      descriptor relation wearing hat
        -> person --wearing--> hat
    """

    def handle(base_object: str, object_text: str, descs: List[Dict[str, Any]]):
        base_object = clean_phrase(base_object)
        object_text = clean_phrase(object_text) or base_object

        if not base_object:
            return

        for d in descs or []:
            dtype = clean_phrase(d.get("type", ""))

            if dtype == "owner":
                owner = clean_phrase(d.get("value", ""))
                predicate = clean_phrase(d.get("relation", "")) or infer_owner_relation(base_object)

                if owner:
                    add_object_once(source_objects, owner)

                    add_relation_once(
                        relations,
                        RelationSpec(
                            subject=owner,
                            predicate=predicate,
                            object=base_object,
                            evidence=object_text,
                        )
                    )

            elif dtype == "relation":
                subject = base_object
                predicate = clean_phrase(d.get("relation", ""))
                obj = clean_phrase(d.get("object", ""))

                if subject and predicate and obj:
                    add_object_once(source_objects, subject)
                    add_object_once(source_objects, obj)

                    add_relation_once(
                        relations,
                        RelationSpec(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            evidence=object_text,
                        )
                    )

    handle(unit.source_object, unit.source_object_text, unit.source_descriptors)
    handle(unit.target_object, unit.target_object_text, unit.target_descriptors)
    handle(unit.anchor_object, unit.anchor_object_text, unit.anchor_descriptors)


# ============================================================
# PLE annotation parsing
# ============================================================
def extract_bracket_targets(target_prompt: str) -> List[str]:
    targets = re.findall(r"\[([^\]]+)\]", target_prompt or "")
    return [clean_phrase(t) for t in targets if clean_phrase(t)]


def parse_edit_units(record: Dict[str, Any]) -> List[EditUnit]:
    """
    Parse original PLE annotation JSON edit_action field.

    Important:
      edit_action may be either dict or JSON string.
    """
    edit_action = safe_json_loads(record.get("edit_action"), {})
    if not isinstance(edit_action, dict):
        edit_action = {}

    units = []

    for raw_target, info in edit_action.items():
        if not isinstance(info, dict):
            continue

        raw_target = clean_phrase(raw_target)
        raw_source = clean_phrase(info.get("action", raw_target))
        edit_type = int(info.get("edit_type", -1))

        source_base, source_text, source_desc = parse_described_object(raw_source)
        target_base, target_text, target_desc = parse_described_object(raw_target)

        units.append(
            EditUnit(
                target_object=target_base,
                source_object=source_base,
                edit_type=edit_type,
                position=int(info.get("position", -1)),
                operation=EDIT_TYPE_ID_TO_NAME.get(edit_type, ""),
                source_object_text=source_text,
                target_object_text=target_text,
                source_descriptors=source_desc,
                target_descriptors=target_desc,
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

def extract_explicit_relation_phrases(prompt: str) -> List[RelationSpec]:
    """
    Extract explicit relation phrases directly from source_prompt.

    Examples:
      a cat wearing a pink hat
        -> cat --wearing--> hat

      a man holding a red umbrella
        -> man --holding--> umbrella

      a dog with a blue collar
        -> dog --with--> collar
    """
    prompt_norm = clean_phrase(prompt)
    if not prompt_norm:
        return []

    relations: List[RelationSpec] = []

    # Remove leading global article.
    text = re.sub(r"^(a|an|the)\s+", "", prompt_norm)

    relation_alt = r"wearing|holding|with|carrying"

    m = re.search(
        rf"(?P<subject>.+?)\s+(?P<predicate>{relation_alt})\s+(?P<object>.+)",
        text,
    )

    if not m:
        return []

    subject_phrase = clean_phrase(m.group("subject"))
    predicate = clean_phrase(m.group("predicate"))
    object_phrase = clean_phrase(m.group("object"))

    subject_base, subject_text, _ = parse_described_object(subject_phrase)
    object_base, object_text, _ = parse_described_object(object_phrase)

    if subject_base and object_base:
        relations.append(
            RelationSpec(
                subject=subject_base,
                predicate=predicate,
                object=object_base,
                evidence=f"{predicate} {object_text}",
            )
        )

    return relations

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

    edit_units = parse_edit_units(record)

    source_objects = unique_keep_order([u.source_object for u in edit_units if u.source_object])
    target_objects = unique_keep_order([u.target_object for u in edit_units if u.target_object])

    bracket_targets = extract_bracket_targets(target_prompt)
    for t in bracket_targets:
        base, _, _ = parse_described_object(t)
        if base and base not in target_objects:
            target_objects.append(base)

    # Extract explicit source prompt relations:
    # a cat wearing a pink hat -> cat --wearing--> hat
    relations: List[RelationSpec] = []

    # Direct relation extraction:
    # a cat wearing a pink hat -> cat --wearing--> hat
    for r in extract_explicit_relation_phrases(source_prompt):
        add_relation_once(relations, r)

    # Pairwise fallback:
    # cat left of dog / cup on table
    for r in extract_relations_from_prompt(
        source_prompt,
        source_objects + target_objects,
    ):
        add_relation_once(relations, r)

    # Extract descriptor-derived relations:
    # left person's shirt -> person --wearing--> shirt
    for u in edit_units:
        add_descriptor_relations_for_unit(
            unit=u,
            relations=relations,
            source_objects=source_objects,
        )

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
# Command parsing
# ============================================================
def classify_attribute_value(value: str) -> str:
    v = clean_phrase(value)
    words = set(v.split())

    if any(w in words for w in COLOR_WORDS):
        return "color"

    if any(w in words for w in MATERIAL_WORDS):
        return "material"

    return ""


def parse_absolute_position_phrase(p: str) -> Tuple[str, str]:
    p = clean_phrase(p)

    m = re.search(
        r"^(?:at|in|on)\s+(?:the\s+)?"
        r"(?P<loc>(?:upper|lower|top|bottom)\s+(?:left|right)|"
        r"left|right|top|bottom|center|middle)"
        r"(?:\s+(?:side|corner|part|area))?"
        r"(?:\s+of\s+(?P<anchor>.+))?$",
        p,
    )

    if not m:
        return "", ""

    loc = clean_phrase(m.group("loc")).replace("middle", "center").replace(" ", "_")
    anchor = clean_phrase(m.group("anchor") or "image")
    return f"at_{loc}", anchor


def parse_position_phrase(position_text: str) -> Tuple[str, str]:
    p = clean_phrase(position_text)

    abs_rel, abs_anchor = parse_absolute_position_phrase(p)
    if abs_rel:
        return abs_rel, abs_anchor

    for pat, rel in POSITION_PATTERNS:
        m = re.search(pat, p)
        if m:
            return rel, clean_phrase(m.group("anchor"))

    return "", ""


def split_object_and_position_for_add(body: str) -> Tuple[str, str]:
    body = clean_phrase(body)
    if not body:
        return "", ""

    starter_alt = "|".join(ADD_POSITION_STARTERS)
    m = re.search(
        rf"^(?P<obj>.+?)\s+(?P<pos>(?:{starter_alt})\s+.+)$",
        body,
        flags=re.I,
    )
    if m:
        return clean_phrase(m.group("obj")), clean_phrase(m.group("pos"))

    m = re.search(
        r"^(?P<obj>.+?)\s+"
        r"(?P<pos>(?:at|in|on)\s+(?:the\s+)?"
        r"(?:upper\s+left|upper\s+right|lower\s+left|lower\s+right|"
        r"top\s+left|top\s+right|bottom\s+left|bottom\s+right|"
        r"left|right|top|bottom|center|middle)"
        r"(?:\s+(?:side|corner|part|area))?"
        r"(?:\s+of\s+.+)?)$",
        body,
        flags=re.I,
    )
    if m:
        return clean_phrase(m.group("obj")), clean_phrase(m.group("pos"))

    return body, ""


def make_edit_unit(
    operation: str,
    source_object: str = "",
    target_object: str = "",
    attribute: str = "",
    value: str = "",
    position_text: str = "",
    raw_command: str = "",
) -> EditUnit:
    operation = clean_phrase(operation)

    if operation == "remove":
        cmd_operation = "remove"
    elif operation == "erase":
        cmd_operation = "delete"
    elif operation == "relation":
        cmd_operation = "move"
    else:
        cmd_operation = operation

    relation, anchor_raw = parse_position_phrase(position_text) if position_text else ("", "")

    source_base, source_text, source_desc = parse_described_object(source_object)
    target_base, target_text, target_desc = parse_described_object(target_object)
    anchor_base, anchor_text, anchor_desc = parse_described_object(anchor_raw)

    return EditUnit(
        source_object=source_base,
        target_object=target_base,
        edit_type=EDIT_TYPE_MAP.get(cmd_operation, -1),
        position=-1,
        operation=cmd_operation,
        attribute=clean_phrase(attribute),
        value=clean_phrase(value),
        position_text=clean_phrase(position_text),
        anchor_object=anchor_base,
        relation=relation,
        raw_command=raw_command,
        source_object_text=source_text,
        target_object_text=target_text,
        anchor_object_text=anchor_text,
        source_descriptors=source_desc,
        target_descriptors=target_desc,
        anchor_descriptors=anchor_desc,
    )


def split_edit_commands(command: str) -> List[str]:
    command = str(command).strip()
    parts = re.split(r"(?:;|\.|\band\s+then\b|\bthen\b)", command, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def parse_single_edit_command(cmd: str) -> Optional[EditUnit]:
    raw = cmd
    cmd = normalize_text(cmd)

    # change the color of car to red
    m = re.search(
        r"^change\s+(?:the\s+)?color\s+of\s+(?P<obj>.+?)\s+(?:to|into)\s+(?P<value>.+)$",
        cmd,
    )
    if m:
        obj = clean_phrase(m.group("obj"))
        return make_edit_unit(
            "color",
            source_object=obj,
            target_object=obj,
            attribute="color",
            value=m.group("value"),
            raw_command=raw,
        )

    # change car color to red / change the car's color to red
    m = re.search(
        r"^change\s+(?:the\s+)?(?P<obj>.+?)(?:'s)?\s+color\s+(?:to|into)\s+(?P<value>.+)$",
        cmd,
    )
    if m:
        obj = clean_phrase(m.group("obj"))
        return make_edit_unit(
            "color",
            source_object=obj,
            target_object=obj,
            attribute="color",
            value=m.group("value"),
            raw_command=raw,
        )

    # make car red
    color_alt = "|".join(map(re.escape, COLOR_WORDS))
    m = re.search(rf"^make\s+(?P<obj>.+?)\s+(?P<value>{color_alt})$", cmd)
    if m:
        obj = clean_phrase(m.group("obj"))
        return make_edit_unit(
            "color",
            source_object=obj,
            target_object=obj,
            attribute="color",
            value=m.group("value"),
            raw_command=raw,
        )

    # change the material of table to wood
    m = re.search(
        r"^change\s+(?:the\s+)?(?:material|texture)\s+of\s+(?P<obj>.+?)\s+(?:to|into)\s+(?P<value>.+)$",
        cmd,
    )
    if m:
        obj = clean_phrase(m.group("obj"))
        return make_edit_unit(
            "material",
            source_object=obj,
            target_object=obj,
            attribute="material",
            value=m.group("value"),
            raw_command=raw,
        )

    # change table material to wood / change the table's material to wood
    m = re.search(
        r"^change\s+(?:the\s+)?(?P<obj>.+?)(?:'s)?\s+(?:material|texture)\s+(?:to|into)\s+(?P<value>.+)$",
        cmd,
    )
    if m:
        obj = clean_phrase(m.group("obj"))
        return make_edit_unit(
            "material",
            source_object=obj,
            target_object=obj,
            attribute="material",
            value=m.group("value"),
            raw_command=raw,
        )

    # make table wooden
    material_alt = "|".join(map(re.escape, MATERIAL_WORDS))
    m = re.search(rf"^make\s+(?P<obj>.+?)\s+(?P<value>{material_alt})$", cmd)
    if m:
        obj = clean_phrase(m.group("obj"))
        return make_edit_unit(
            "material",
            source_object=obj,
            target_object=obj,
            attribute="material",
            value=m.group("value"),
            raw_command=raw,
        )

    # move / relation
    m = re.search(r"^(?:move|relation)\s+(?P<obj>.+?)\s+to\s+(?P<pos>.+)$", cmd)
    if m:
        obj = clean_phrase(m.group("obj"))
        return make_edit_unit(
            "move",
            source_object=obj,
            target_object=obj,
            position_text=m.group("pos"),
            raw_command=raw,
        )

    starter_alt = "|".join(ADD_POSITION_STARTERS)
    m = re.search(rf"^(?:move|relation)\s+(?P<obj>.+?)\s+(?P<pos>(?:{starter_alt})\s+.+)$", cmd)
    if m:
        obj = clean_phrase(m.group("obj"))
        return make_edit_unit(
            "move",
            source_object=obj,
            target_object=obj,
            position_text=m.group("pos"),
            raw_command=raw,
        )

    # replace dog with cat
    m = re.search(r"^replace\s+(?P<src>.+?)\s+with\s+(?P<tgt>.+)$", cmd)
    if m:
        return make_edit_unit(
            "replace",
            source_object=m.group("src"),
            target_object=m.group("tgt"),
            raw_command=raw,
        )

    # change dog to cat
    # change left person's shirt to blue
    m = re.search(r"^change\s+(?P<src>.+?)\s+(?:to|into)\s+(?P<tgt>.+)$", cmd)
    if m:
        src = clean_phrase(m.group("src"))
        tgt = clean_phrase(m.group("tgt"))
        attr = classify_attribute_value(tgt)

        if attr == "color":
            return make_edit_unit(
                "color",
                source_object=src,
                target_object=src,
                attribute="color",
                value=tgt,
                raw_command=raw,
            )

        if attr == "material":
            return make_edit_unit(
                "material",
                source_object=src,
                target_object=src,
                attribute="material",
                value=tgt,
                raw_command=raw,
            )

        return make_edit_unit(
            "replace",
            source_object=src,
            target_object=tgt,
            raw_command=raw,
        )

    # delete / remove / erase
    m = re.search(r"^(?:delete|remove|erase)\s+(?P<obj>.+)$", cmd)
    if m:
        obj = clean_phrase(m.group("obj"))
        if cmd.startswith("remove"):
            op = "remove"
        elif cmd.startswith("erase"):
            op = "delete"
        else:
            op = "delete"

        return make_edit_unit(
            op,
            source_object=obj,
            target_object="",
            raw_command=raw,
        )

    # add
    m = re.search(r"^add\s+(?P<body>.+)$", cmd)
    if m:
        body = clean_phrase(m.group("body"))
        obj, pos = split_object_and_position_for_add(body)

        return make_edit_unit(
            "add",
            source_object="",
            target_object=obj,
            position_text=pos,
            raw_command=raw,
        )

    return None


def parse_edit_command(command: str) -> List[EditUnit]:
    units = []

    for part in split_edit_commands(command):
        unit = parse_single_edit_command(part)
        if unit is not None:
            units.append(unit)
        else:
            print(f"[Rule Parser] warning: cannot parse command fragment: {part}")

    return units


def parse_command_task(
    image_path: str,
    command: str,
    sample_id: Optional[str] = None,
) -> ParsedTask:
    if sample_id is None:
        sample_id = os.path.splitext(os.path.basename(image_path))[0]

    edit_units = parse_edit_command(command)

    source_objects: List[str] = []
    target_objects: List[str] = []
    relations: List[RelationSpec] = []

    for u in edit_units:
        if u.source_object:
            add_object_once(source_objects, u.source_object)

        if u.target_object:
            add_object_once(target_objects, u.target_object)

        # Explicit spatial relation:
        # add cat left of dog -> cat --left_of--> dog
        # move cup on table -> cup --on--> table
        if u.relation and u.anchor_object:
            subject = u.target_object or u.source_object

            add_relation_once(
                relations,
                RelationSpec(
                    subject=subject,
                    predicate=u.relation,
                    object=u.anchor_object,
                    evidence=u.position_text,
                )
            )

            add_object_once(source_objects, u.anchor_object)

        # Descriptor-derived relation:
        # left person's shirt -> person --wearing--> shirt
        add_descriptor_relations_for_unit(
            unit=u,
            relations=relations,
            source_objects=source_objects,
        )

    return ParsedTask(
        sample_id=str(sample_id),
        image_path=image_path,
        source_prompt="",
        target_prompt=command,
        edit_units=edit_units,
        relations=relations,
        source_objects=source_objects,
        target_objects=target_objects,
    )


# ============================================================
# Detection target helper
# ============================================================
def get_detection_targets(task_dict_or_task) -> List[str]:
    """
    Detection targets should be base objects plus anchors and relation nodes.

    Examples:
      remove the left knife              -> ["knife"]
      replace dog with cat               -> ["dog"]
      change left person's shirt to blue -> ["shirt", "person"]
      remove left white dog              -> ["dog"]
      move cat right of dog              -> ["cat", "dog"]
      add cat left of dog                -> ["dog"]
      cat wearing hat                    -> ["cat", "hat"]
    """
    if isinstance(task_dict_or_task, dict):
        edit_units = task_dict_or_task.get("edit_units", [])
        relations = task_dict_or_task.get("relations", [])
    else:
        edit_units = getattr(task_dict_or_task, "edit_units", [])
        relations = getattr(task_dict_or_task, "relations", [])

    targets: List[str] = []

    def add_target(x: str):
        x = clean_phrase(x)
        if x and x not in ["image", "scene"] and x not in targets:
            targets.append(x)

    for u in edit_units:
        if isinstance(u, dict):
            op = clean_phrase(u.get("operation", ""))
            edit_type = int(u.get("edit_type", -1))
            src = clean_phrase(u.get("source_object", ""))
            anchor = clean_phrase(u.get("anchor_object", ""))
        else:
            op = clean_phrase(getattr(u, "operation", ""))
            edit_type = int(getattr(u, "edit_type", -1))
            src = clean_phrase(getattr(u, "source_object", ""))
            anchor = clean_phrase(getattr(u, "anchor_object", ""))

        if op in ["delete", "remove", "replace", "move", "color", "material"] or edit_type in [1, 2, 3, 4, 5]:
            add_target(src)

        if op in ["add", "move"] or edit_type in [0, 3]:
            add_target(anchor)

    # Objects introduced by ADD do not exist in the source image.
    # Keep their relation for placement reasoning, but never query DINO for them.
    virtual_add_targets = set()
    for u in edit_units:
        if isinstance(u, dict):
            op = clean_phrase(u.get("operation", ""))
            edit_type = int(u.get("edit_type", -1))
            target = clean_phrase(u.get("target_object", ""))
        else:
            op = clean_phrase(getattr(u, "operation", ""))
            edit_type = int(getattr(u, "edit_type", -1))
            target = clean_phrase(getattr(u, "target_object", ""))

        if (op == "add" or edit_type == 0) and target:
            virtual_add_targets.add(target)

    for r in relations:
        if isinstance(r, dict):
            subject = clean_phrase(r.get("subject", ""))
            obj = clean_phrase(r.get("object", ""))
        else:
            subject = clean_phrase(getattr(r, "subject", ""))
            obj = clean_phrase(getattr(r, "object", ""))

        if subject not in virtual_add_targets:
            add_target(subject)
        if obj not in virtual_add_targets:
            add_target(obj)

    return targets


# ============================================================
# Output / visualization
# ============================================================
def is_ple_annotation_file(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False

    if not path.lower().endswith(".json"):
        return False

    try:
        records = load_annotations(path)
    except Exception:
        return False

    if not records:
        return False

    r = records[0]
    if not isinstance(r, dict):
        return False

    required = ["image", "source_prompt", "target_prompt", "edit_action"]
    return all(k in r for k in required)


def infer_ple_output_root(input_json: str) -> str:
    """
    Example:
      data/PLE_bench/0_random_140/export/annotations.json
    ->
      outputs/PLE_bench/0_random_140
    """
    parts = os.path.normpath(input_json).split(os.sep)

    if "PLE_bench" in parts:
        i = parts.index("PLE_bench")
        if i + 1 < len(parts):
            return os.path.join("outputs", "PLE_bench", parts[i + 1])

    return os.path.join("outputs", "PLE_bench")


def resolve_json_input(input_json: str):
    ann_path = os.path.abspath(input_json)
    out_dir = infer_ple_output_root(input_json)
    subset_name = os.path.splitext(os.path.basename(ann_path))[0]
    return ann_path, out_dir, subset_name


def get_font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def visualize_parsed_task(task: ParsedTask, out_path: str):
    font = get_font(18)
    line_h = 30
    lines = [
        f"id: {task.sample_id}",
        f"image: {task.image_path}",
        f"source_prompt: {task.source_prompt}",
        f"target_prompt: {task.target_prompt}",
        "",
        f"source_objects for detection: {task.source_objects}",
        f"target_objects for editing: {task.target_objects}",
        "",
        "edit_units:",
    ]

    for u in task.edit_units:
        extra = []
        if u.operation:
            extra.append(f"operation={u.operation}")
        if u.attribute:
            extra.append(f"attribute={u.attribute}")
        if u.value:
            extra.append(f"value={u.value}")
        if u.relation:
            extra.append(f"relation={u.relation}")
        if u.anchor_object:
            extra.append(f"anchor={u.anchor_object}")
        if u.position_text:
            extra.append(f"position_text={u.position_text}")
        if u.source_object_text:
            extra.append(f"source_text={u.source_object_text}")
        if u.target_object_text:
            extra.append(f"target_text={u.target_object_text}")
        if u.anchor_object_text:
            extra.append(f"anchor_text={u.anchor_object_text}")
        if u.source_descriptors:
            extra.append(f"source_desc={u.source_descriptors}")
        if u.target_descriptors:
            extra.append(f"target_desc={u.target_descriptors}")
        if u.anchor_descriptors:
            extra.append(f"anchor_desc={u.anchor_descriptors}")

        lines.append(
            f"  source={u.source_object} -> target={u.target_object}, "
            f"edit_type={u.edit_type}, position={u.position}"
            + (f", {', '.join(extra)}" if extra else "")
        )

    lines.append("")
    lines.append("relations:")

    if task.relations:
        for r in task.relations:
            lines.append(f"  {r.subject} --{r.predicate}--> {r.object}, evidence='{r.evidence}'")
    else:
        lines.append("  no explicit relation parsed")

    width = 1800
    height = max(300, 40 + line_h * len(lines))

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        y += line_h

    ensure_dir(os.path.dirname(out_path))
    img.save(out_path)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "ple", "cmd"])

    # PLE input
    parser.add_argument("--ann-rel", type=str, default="0_random_140/export/annotations.json")
    parser.add_argument("--input-json", type=str, default="")
    parser.add_argument("--idx", type=int, default=0)

    # CMD input
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--cmd", type=str, default="")
    parser.add_argument("--sample-id", type=str, default="")

    # Optional override
    parser.add_argument("--out-dir", type=str, default="")

    args = parser.parse_args()

    use_cmd_mode = args.mode == "cmd" or (
        args.mode == "auto" and args.image and args.cmd
    )

    # --------------------------------------------------------
    # Mode 1: command
    # --------------------------------------------------------
    if use_cmd_mode:
        if not args.image:
            raise ValueError("cmd mode requires --image")
        if not args.cmd:
            raise ValueError("cmd mode requires --cmd")

        task = parse_command_task(
            image_path=args.image,
            command=args.cmd,
            sample_id=args.sample_id or None,
        )

        out_dir = args.out_dir or os.path.join("outputs", "command")
        sample_dir = get_sample_out_dir(out_dir, task.sample_id)

        save_json(task.to_dict(), os.path.join(sample_dir, "00_parsed_task.json"))
        visualize_parsed_task(task, os.path.join(sample_dir, "00_parsed_task.jpg"))

        print(f"[Rule Parser CMD] saved to {sample_dir}")
        print(json.dumps(task.to_dict(), indent=2, ensure_ascii=False))
        return

    # --------------------------------------------------------
    # Mode 2: PLE annotation
    # --------------------------------------------------------
    if args.input_json:
        ann_path, out_dir, subset_name = resolve_json_input(args.input_json)
    else:
        ann_path, out_dir, subset_name = resolve_ann_and_out(args.ann_rel)

    if args.out_dir:
        out_dir = args.out_dir

    if args.mode in ["auto", "ple"]:
        if not is_ple_annotation_file(ann_path):
            raise ValueError(
                f"Input json does not look like a PLE annotation file: {ann_path}. "
                f"Expected keys: image/source_prompt/target_prompt/edit_action. "
                f"If you want command mode, use --image and --cmd."
            )

    records = load_annotations(ann_path)

    if args.idx < 0 or args.idx >= len(records):
        raise IndexError(f"--idx {args.idx} out of range, total records={len(records)}")

    record = records[args.idx]
    task = parse_annotation_record(record)

    sample_dir = get_sample_out_dir(out_dir, task.sample_id)

    save_json(task.to_dict(), os.path.join(sample_dir, "00_parsed_task.json"))
    visualize_parsed_task(task, os.path.join(sample_dir, "00_parsed_task.jpg"))

    print(f"[Rule Parser PLE] saved to {sample_dir}")
    print(json.dumps(task.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
