# description_generator.py
"""
Non-recursive, cleaned-up graph-aware description generator.

Provides:
 - generate_description_from_entry(entry) -> str
 - refine_description(entry, previous_description, user_instruction) -> str
 - compute_node_salience(entry) -> (scores_list, obj_ids, method)
 - get_cleaned_entry(entry) -> cleaned_entry (merged parts, deduped)
 - get_runtime_info(entry) -> dict

Lightweight, deterministic; no torch dependency required.
"""

from typing import Dict, List, Tuple, Optional
import json, math, re

# ----------------------
# Basic geometry & naming helpers
# ----------------------
def _bbox(o: Dict) -> Tuple[float,float,float,float]:
    x = float(o.get("x", 0)); y = float(o.get("y", 0))
    w = float(o.get("w", 0)); h = float(o.get("h", 0))
    return (x, y, x + w, y + h)

def _area(o: Dict) -> float:
    x1,y1,x2,y2 = _bbox(o)
    return max(0.0, (x2-x1)*(y2-y1))

def _center(o: Dict) -> Tuple[float,float]:
    x1,y1,x2,y2 = _bbox(o)
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def _iou(a: Dict, b: Dict) -> float:
    ax1,ay1,ax2,ay2 = _bbox(a)
    bx1,by1,bx2,by2 = _bbox(b)
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    iw = max(0.0, ix2-ix1); ih = max(0.0, iy2-iy1)
    inter = iw*ih
    area_a = max(0.0, (ax2-ax1)*(ay2-ay1))
    area_b = max(0.0, (bx2-bx1)*(by2-by1))
    uni = area_a + area_b - inter
    return inter/uni if uni > 0 else 0.0

def _norm_name(o: Dict) -> str:
    names = o.get("names") or []
    if names and len(names)>0:
        return names[0].replace("_"," ").strip().lower()
    syns = o.get("synsets") or []
    if syns and len(syns)>0:
        return syns[0].split(".")[0].replace("_"," ").strip().lower()
    return "object"

# ----------------------
# Coarse label mapping
# ----------------------
COARSE_MAP = {
    "person":"person","man":"person","woman":"person","guy":"person","boy":"person",
    "bicycle":"bicycle","bike":"bicycle","bikes":"bicycle",
    "car":"vehicle","truck":"vehicle","van":"vehicle","work truck":"vehicle","taxi":"vehicle",
    "building":"building","wall":"building","window":"building","windows":"building",
    "tree":"tree","trees":"tree","tree trunk":"tree",
    "sidewalk":"infrastructure","street":"infrastructure","road":"infrastructure",
    "parking meter":"infrastructure","lamp post":"infrastructure","lamp":"infrastructure","bench":"infrastructure"
}
PART_KEYWORDS = {"shoe","shoes","sneaker","sneakers","chin","arm","back","glasses","jacket","shirt","pants","trouser","headlight"}

def _coarse_label(name: str) -> str:
    n = name.lower()
    if n in COARSE_MAP:
        return COARSE_MAP[n]
    for k,v in COARSE_MAP.items():
        if k in n:
            return v
    for p in PART_KEYWORDS:
        if p in n:
            return "part"
    if any(tok in n for tok in ["building","house","store","shop"]):
        return "building"
    return "object"

# ----------------------
# Collect relations safely
# ----------------------
def _collect_relations(entry: Dict) -> List[Tuple[int,str,int]]:
    rels = []
    for r in entry.get("relationships", []):
        try:
            sid = int(r.get("subject_id"))
            oid = int(r.get("object_id"))
            pred = (r.get("predicate") or "").strip()
            if pred:
                rels.append((sid, pred, oid))
        except Exception:
            continue
    return rels

# ----------------------
# Merge small "parts" into parent objects (wearing/holding/windows)
# ----------------------
def _merge_parts_and_windows(entry: Dict, area_thresh_ratio: float = 0.02) -> Dict:
    cleaned = json.loads(json.dumps(entry))
    objs = cleaned.get("objects", [])
    if not objs:
        return cleaned
    id2obj = {int(o["object_id"]): o for o in objs}
    remove_ids = set()

    # attach objects referenced by wearing/holding relations to subject
    attach_preds = {"wears","wearing","wear","holding","hold","has","have","wea ring","WEARING"}
    for r in entry.get("relationships", []):
        pred = (r.get("predicate") or "").lower()
        try:
            sid = int(r.get("subject_id")); oid = int(r.get("object_id"))
        except Exception:
            continue
        if pred and any(k in pred for k in attach_preds) and sid in id2obj and oid in id2obj:
            subj = id2obj[sid]; obj = id2obj[oid]
            subj_attrs = subj.get("attributes") or []
            oname = _norm_name(obj)
            # merge object's attributes and name as attribute of subject
            for a in (obj.get("attributes") or []):
                if a not in subj_attrs:
                    subj_attrs.append(a)
            if oname and oname not in subj_attrs:
                subj_attrs.append(oname)
            subj["attributes"] = subj_attrs
            remove_ids.add(oid)

    # attach windows to nearest building if overlapping
    for oid, o in list(id2obj.items()):
        if oid in remove_ids:
            continue
        name = _norm_name(o)
        if "window" in name:
            best_bid = None; best_iou = 0.0
            for bid, b in id2obj.items():
                if bid == oid or bid in remove_ids: continue
                if "building" not in _coarse_label(_norm_name(b)):
                    continue
                i = _iou(o, b)
                if i > best_iou:
                    best_iou = i; best_bid = bid
            if best_bid is not None and best_iou > 0.04:
                b = id2obj[best_bid]
                b_attrs = b.get("attributes") or []
                if "windows" not in b_attrs:
                    b_attrs.append("windows")
                b["attributes"] = b_attrs
                remove_ids.add(oid)

    # merge tiny parts into nearest parent (person/vehicle/building) if inside bbox
    xs = [float(o.get("x",0)) for o in objs]; ws = [float(o.get("w",0)) for o in objs]
    ys = [float(o.get("y",0)) for o in objs]; hs = [float(o.get("h",0)) for o in objs]
    max_x = max([x+w for x,w in zip(xs,ws)]) if xs else 1.0
    max_y = max([y+h for y,h in zip(ys,hs)]) if ys else 1.0
    img_area = max_x * max_y if max_x>0 and max_y>0 else 1.0

    for oid, o in list(id2obj.items()):
        if oid in remove_ids:
            continue
        name = _norm_name(o)
        if _coarse_label(name) != "part":
            continue
        a = _area(o)
        if a < area_thresh_ratio * img_area:
            cx, cy = _center(o)
            best_pid = None; best_dist = float("inf")
            for pid, p in id2obj.items():
                if pid == oid or pid in remove_ids: continue
                if _coarse_label(_norm_name(p)) not in ("person","vehicle","building"):
                    continue
                x1,y1,x2,y2 = _bbox(p)
                if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                    continue
                px, py = _center(p)
                dist = (px-cx)**2 + (py-cy)**2
                if dist < best_dist:
                    best_dist = dist; best_pid = pid
            if best_pid:
                parent = id2obj[best_pid]
                p_attrs = parent.get("attributes") or []
                for at in (o.get("attributes") or []):
                    if at not in p_attrs:
                        p_attrs.append(at)
                oname = _norm_name(o)
                if oname not in p_attrs and len(oname) < 24:
                    p_attrs.append(oname)
                parent["attributes"] = p_attrs
                remove_ids.add(oid)

    new_objs = [o for o in objs if int(o.get("object_id")) not in remove_ids]
    new_rels = []
    for r in entry.get("relationships", []):
        try:
            sid = int(r.get("subject_id")); oid = int(r.get("object_id"))
            if sid in remove_ids or oid in remove_ids:
                continue
            new_rels.append(r)
        except Exception:
            continue

    cleaned["objects"] = new_objs
    cleaned["relationships"] = new_rels
    return cleaned

# ----------------------
# Deduplication (merge overlapping objects of same coarse type)
# ----------------------
def _dedupe_objects(entry: Dict, iou_thresh: float = 0.6, attr_merge_iou: float = 0.8) -> Dict:
    cleaned = json.loads(json.dumps(entry))
    objs = cleaned.get("objects", [])
    keep = []
    used = set()
    for i, o in enumerate(objs):
        if i in used: continue
        group = [o]
        for j in range(i+1, len(objs)):
            if j in used: continue
            if _coarse_label(_norm_name(o)) != _coarse_label(_norm_name(objs[j])):
                continue
            if _iou(o, objs[j]) >= iou_thresh:
                group.append(objs[j]); used.add(j)
        rep = max(group, key=lambda x: _area(x))
        rep_attrs = rep.get("attributes") or []
        for m in group:
            if m is rep: continue
            if _iou(m, rep) >= attr_merge_iou:
                for a in (m.get("attributes") or []):
                    if a not in rep_attrs:
                        rep_attrs.append(a)
        rep["attributes"] = rep_attrs
        keep.append(rep)
    cleaned["objects"] = keep
    valid_ids = {int(o["object_id"]) for o in keep}
    new_rels = []
    for r in cleaned.get("relationships", []):
        try:
            sid = int(r.get("subject_id")); oid = int(r.get("object_id"))
            if sid in valid_ids and oid in valid_ids:
                new_rels.append(r)
        except Exception:
            continue
    cleaned["relationships"] = new_rels
    return cleaned

# ----------------------
# Hybrid salience: area + degree in relation graph
# ----------------------
def compute_node_salience(entry: Dict) -> Tuple[List[float], List[int], str]:
    objs = entry.get("objects", [])
    if not objs:
        return [], [], "area"
    areas = [_area(o) for o in objs]
    max_a = max(areas) if areas else 1.0
    id2idx = {int(o["object_id"]): idx for idx,o in enumerate(objs)}
    deg = [0]*len(objs)
    for sid,_,oid in _collect_relations(entry):
        if sid in id2idx: deg[id2idx[sid]] += 1
        if oid in id2idx: deg[id2idx[oid]] += 1
    max_deg = max(deg) if deg else 1
    norm_areas = [a/max_a for a in areas]
    norm_deg = [d/max_deg for d in deg]
    scores = [na + 0.5*nd for na,nd in zip(norm_areas, norm_deg)]
    maxs = max(scores) if scores else 1.0
    scores = [s/maxs for s in scores]
    obj_ids = [int(o["object_id"]) for o in objs]
    return scores, obj_ids, "hybrid"

# ----------------------
# Serialize cleaned facts (single non-recursive function)
# ----------------------
def _serialize_cleaned_facts(entry: Dict, top_k_rels: int = 8):
    merged = _merge_parts_and_windows(entry)
    cleaned = _dedupe_objects(merged)
    objs = cleaned.get("objects", [])
    rels = _collect_relations(cleaned)
    scores, obj_ids, method = compute_node_salience(cleaned)
    id2score = {oid: sc for oid, sc in zip(obj_ids, scores)} if obj_ids else {}
    id2obj = {int(o["object_id"]): o for o in objs}

    ordered_ids = sorted([int(o["object_id"]) for o in objs], key=lambda i: -id2score.get(i,0.0))
    facts_objs = []
    for oid in ordered_ids:
        o = id2obj.get(oid)
        if not o: 
            continue
        name = _norm_name(o)
        coarse = _coarse_label(name)
        if coarse == "part":
            continue
        # Keep display as the canonical name only (do not embed small attributes).
        display = name
        facts_objs.append((oid, display, coarse, id2score.get(oid, 0.0)))

    def rel_score(t):
        sid,pred,oid = t
        s = id2score.get(int(sid), 0.0); o_ = id2score.get(int(oid), 0.0)
        pred_w = 2.0 if pred.lower() not in ("near","on","in","") else 0.5
        return s + o_ + pred_w
    rels_sorted = sorted(rels, key=rel_score, reverse=True)[:top_k_rels]

    scene_attrs = entry.get("scene_attributes", {}) or {}
    scene_desc = []
    if scene_attrs.get("time_of_day"): scene_desc.append(scene_attrs["time_of_day"])
    if scene_attrs.get("weather"): scene_desc.append(scene_attrs["weather"])
    return cleaned, facts_objs, rels_sorted, scene_desc, id2score, method

# ----------------------
# Realization: compose a coherent paragraph (deterministic)
# ----------------------
def _pluralize_label(label: str, n: int) -> str:
    if label == "person":
        return f"{n} pedestrian{'s' if n!=1 else ''}"
    if label == "bicycle":
        return f"{n} bike{'s' if n!=1 else ''}"
    if label == "vehicle":
        return f"{n} vehicle{'s' if n!=1 else ''}"
    return f"{n} {label}{'s' if n!=1 else ''}"

def _safe_prefix(display: str, count: int = 1) -> str:
    last = display.split()[-1] if display else ""
    if count > 1 or last.endswith("s"):
        return display
    if display.startswith(("a ","an ","the ")):
        return display
    return "a " + display

def _format_person_clothing(o: Dict) -> Optional[str]:
    attrs = o.get("attributes") or []
    garments = {}
    colors = []
    for a in attrs:
        al = a.lower()
        m = re.match(r"(red|blue|grey|gray|black|white|brown|orange)\s+(\w+)", al)
        if m:
            c,g = m.group(1), m.group(2)
            garments.setdefault(g, []).append(c)
            continue
        if al in ("red","blue","grey","gray","black","white","brown","orange"):
            colors.append(al)
        if al in ("shirt","jacket","pants","trousers","sneakers","shoes"):
            garments.setdefault(al, [])
    parts = []
    for g, cols in garments.items():
        if cols:
            parts.append(f"{' and '.join(cols)} {g}")
        else:
            parts.append(g)
    if not parts and colors:
        parts.append("clothing in " + " and ".join(colors))
    if parts:
        return "wearing " + ", ".join(parts)
    return None

# --- small helper additions for better English ---
def _article_for(word: str) -> str:
    """Return 'a' or 'an' for a surface word using a simple vowel heuristic."""
    if not word:
        return "a"
    w = word.strip().lower()
    if w[0] in "aeiou":
        return "an"
    return "a"

def _human_join(items: List[str]) -> str:
    """Join a list into 'x, y and z' form (Oxford comma optional)."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return items[0] + " and " + items[1]
    return ", ".join(items[:-1]) + ", and " + items[-1]

def _clean_display(s: str) -> str:
    return s.replace("  ", " ").strip()

# improved clothing formatting (slight modification of previous)
def _format_person_clothing(o: Dict) -> Optional[str]:
    attrs = o.get("attributes") or []
    garments = {}
    colors = []
    for a in attrs:
        al = a.lower().strip()
        m = re.match(r"(red|blue|grey|gray|black|white|brown|orange|green)\s+(\w+)", al)
        if m:
            c,g = m.group(1), m.group(2)
            garments.setdefault(g, []).append(c)
            continue
        if al in ("red","blue","grey","gray","black","white","brown","orange","green"):
            colors.append(al)
        if al in ("shirt","jacket","pants","trousers","sneakers","shoes"):
            garments.setdefault(al, [])
    parts = []
    for g, cols in garments.items():
        if cols:
            parts.append(f"{_human_join(cols)} {g}")
        else:
            parts.append(g)
    if not parts and colors:
        parts.append("clothing in " + _human_join(colors))
    if parts:
        return "wearing " + _human_join(parts)
    return None

def _better_pluralize_label(label: str, n: int) -> str:
    # nicer labels for counts
    if label in ("person", "people", "man", "guy"):
        return f"{n} pedestrian{'s' if n != 1 else ''}"
    if label in ("bicycle", "bike", "bikes"):
        return f"{n} bike{'s' if n != 1 else ''}"
    if label == "vehicle":
        return f"{n} vehicle{'s' if n != 1 else ''}"
    if label == "building":
        return f"{n} building{'s' if n != 1 else ''}"
    if label == "tree":
        return f"{n} tree{'s' if n != 1 else ''}"
    if label == "infrastructure":
        return f"{n} infrastructure element{'s' if n != 1 else ''}"
    return f"{n} {label}{'s' if n != 1 else ''}"

# --- main improved realization using display names ---

# -------------------------
# Minor-tuning helpers (normalize, prune, humanize)
# -------------------------
def _normalize_display_name(display: str) -> str:
    """Apply small normalizations to display names to avoid placeholders and repeated words."""
    if not display:
        return display
    d = display.strip()
    # normalize common forms
    d = re.sub(r"\bside ?walk\b", "sidewalk", d, flags=re.I)
    d = re.sub(r"\blamp post\b", "lamp post", d, flags=re.I)
    d = re.sub(r"\bparking meter\b", "parking meter", d, flags=re.I)
    d = re.sub(r"\bwork truck\b", "work truck", d, flags=re.I)
    d = re.sub(r"\bwindow(s)?\b", "windows", d, flags=re.I)
    d = re.sub(r"\bsign\b", "sign", d, flags=re.I)
    # collapse duplicated adjacent words "sidewalk sidewalk" -> "sidewalk"
    d = re.sub(r"\b(\w+)(?: \1\b)+", r"\1", d, flags=re.I)
    return d

# ---------- plural/article helpers ----------
def _is_plural_surface(s: str) -> bool:
    """Rudimentary plural detector for surface display tokens."""
    if not s:
        return False
    s = s.strip().lower()
    # treat obvious plurals ending with s (but ignore short words like 'us', 'as')
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        return True
    # some known group tokens
    if s in ("trees","windows","bikes","people"):
        return True
    return False

def _singularize_surface(s: str) -> str:
    """Naive singularizer for display tokens (used only for article insertion)."""
    if _is_plural_surface(s):
        return s[:-1]
    return s

def _format_with_article_for_surface(display: str) -> str:
    """Return either 'a X' or 'X' (no article) depending on plural detection."""
    if not display:
        return ""
    if _is_plural_surface(display):
        return display  # plural — do not prefix with 'a/an'
    return f"{_article_for(display)} {display}"

def _prune_relations(rels_sorted: List[Tuple[int,str,int]], cleaned_objs: List[Dict], max_rels: int = 6):
    """
    Prune low-value or duplicate relations from rels_sorted.
    Keeps up to max_rels, prefers informative predicates and removes relations involving 'part' nodes.
    """
    def _get_obj(oid):
        return next((x for x in cleaned_objs if int(x["object_id"]) == int(oid)), None)

    seen = set()
    out = []
    for sid, pred, oid in rels_sorted:
        subj = _get_obj(sid); obj = _get_obj(oid)
        if not subj or not obj:
            continue
        # skip if either is a 'part' (we merge parts earlier)
        if _coarse_label(_norm_name(subj)) == "part" or _coarse_label(_norm_name(obj)) == "part":
            continue
        pred_norm = (pred or "").strip().lower()
        subj_disp = _normalize_display_name(_norm_name(subj))
        obj_disp = _normalize_display_name(_norm_name(obj))
        key = (pred_norm, subj_disp, obj_disp)
        if key in seen:
            continue
        # deprioritize extremely generic relations if we already have enough
        if pred_norm in ("has","have","contain","holding") and len(out) >= max_rels:
            continue
        seen.add(key)
        out.append((sid, pred, oid))
        if len(out) >= max_rels:
            break
    return out

def _normalize_attribute_phrase(coarse: str, attrs: List[str], display: str) -> Optional[str]:
    """
    Build a short human-friendly attribute phrase for an object display string.
    Returns None if nothing useful.
    """
    if not attrs:
        return None
    norm_attrs = [a.strip() for a in attrs if a and isinstance(a, str)]
    if not norm_attrs:
        return None
    # vehicle headlights off -> produce explicit phrase
    if coarse == "vehicle" and any(a.lower() == "off" for a in norm_attrs):
        return f"{display} with headlights off"
    # parked
    if "parked" in [a.lower() for a in norm_attrs]:
        return f"{display} that is parked"
    # chained/locked bikes
    if coarse == "bicycle" and any("chained" in a.lower() or "locked" in a.lower() for a in norm_attrs):
        return f"{display} that is chained"
    # parking meter color or small detail
    if "parking meter" in display and norm_attrs:
        return f"{display} that is {norm_attrs[0]}"
    # otherwise show up to two concise attrs not duplicating display term
    chosen = []
    for a in norm_attrs:
        if a.lower() in display.lower():
            continue
        if a not in chosen:
            chosen.append(a)
        if len(chosen) >= 2:
            break
    if chosen:
        return f"{display} that is {_human_join(chosen)}"
    return None


def generate_description_from_entry(entry: Dict, top_k_layout: int = 3, max_relations: int = 6) -> str:
    """
    Deterministic description generator (minor tuned):
     - uses name-only display tokens
     - prunes/normalizes relations
     - builds attribute phrases without duplication
     - avoids counting generic 'object' and renames 'infrastructure' to 'street elements'
    """
    cleaned, facts_objs, rels_sorted, scene_desc, id2score, method = _serialize_cleaned_facts(entry, top_k_rels=24)
    cleaned_objs = cleaned.get("objects", []) if cleaned else []

    # Normalize display names in facts list (but keeps display = name)
    facts_objs = [(oid, _normalize_display_name(display), coarse, sc) for (oid, display, coarse, sc) in facts_objs]

    # Prune relations (use the helper you already added)
    rels_sorted = _prune_relations(rels_sorted, cleaned_objs, max_rels=max_relations)

    # Build id->display mapping and keep attribute lists separate
    id2display = {}
    for oid, display, coarse, score in facts_objs:
        id2display[int(oid)] = display
    for o in cleaned_objs:
        oid = int(o["object_id"])
        if oid not in id2display:
            id2display[oid] = _normalize_display_name(_norm_name(o))

    parts = []
    # Scene overview
    if scene_desc:
        parts.append(f"It appears to be {', '.join(scene_desc)} in this urban street scene.")
    else:
        parts.append("This is an urban street scene.")

    # Counts: filter out generic 'object' and rename 'infrastructure' -> 'street elements'
    counts = {}
    for o in cleaned_objs:
        coarse = _coarse_label(_norm_name(o))
        if coarse in ("part", "object"):
            continue
        counts[coarse] = counts.get(coarse, 0) + 1
    if counts:
        # limit to top 4 categories by count
        top_counts = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        count_phrases = []
        for k, v in top_counts:
            label = k
            if k == "infrastructure":
                label = "street element"
            count_phrases.append(_better_pluralize_label(label, v))
        parts.append("Notably, the scene contains " + _human_join(count_phrases) + ".")

    # Layout: pick top_k_layout by salience then order left->right
    layout_items = []
    for oid, display, coarse, score in facts_objs:
        if coarse == "part":
            continue
        o = next((x for x in cleaned_objs if int(x["object_id"]) == int(oid)), None)
        if not o:
            continue
        layout_items.append((oid, display, coarse))
        if len(layout_items) >= top_k_layout:
            break
    if layout_items:
        def cx_key(t):
            o = next((x for x in cleaned_objs if int(x["object_id"]) == int(t[0])), None)
            return _center(o)[0] if o else 0
        ordered = sorted(layout_items, key=cx_key)
        phrases = []
        for oid, display, coarse in ordered:
            display = _clean_display(display)
            # persons get clothing phrasing
            if coarse == "person":
                o = next((x for x in cleaned_objs if int(x["object_id"]) == int(oid)), None)
                clothing = _format_person_clothing(o) if o else None
                if clothing:
                    phrases.append(f"{_article_for('person')} person {clothing}")
                else:
                    phrases.append(_format_with_article_for_surface(display))
            else:
                phrases.append(_format_with_article_for_surface(display))
        # join with commas and an 'and' for natural reading
        pretty = _human_join(phrases)
        parts.append("Visually, one can see " + pretty + ".")

    # Relations: grouped and synthesized
    grouped = {}
    for sid, pred, oid in rels_sorted:
        subj = next((x for x in cleaned_objs if int(x["object_id"]) == int(sid)), None)
        obj = next((x for x in cleaned_objs if int(x["object_id"]) == int(oid)), None)
        if not subj or not obj:
            continue
        pred_l = (pred or "").strip().lower()
        obj_disp = id2display.get(int(oid), _normalize_display_name(_norm_name(obj)))
        subj_disp = id2display.get(int(sid), _normalize_display_name(_norm_name(subj)))
        key = (pred_l, obj_disp)
        grouped.setdefault(key, set()).add(subj_disp)

    rel_sentences = []
    for (pred_l, obj_disp), subj_set in grouped.items():
        subj_list = sorted(list(subj_set))
        # choose subject phrase naturally, and pick correct verb agreement
        subj_list_simple = []
        for sname in subj_list:
            # if the subject surface (sname) is plural, keep as-is; else add article
            if _is_plural_surface(sname):
                subj_list_simple.append(sname)
            else:
                subj_list_simple.append(f"{_article_for(sname)} {sname}")

        # Determine subject phrase and correct verb: single-plural agreement handled
        if len(subj_list_simple) == 1:
            subj_phrase = subj_list_simple[0]
            # If the raw subj display (without article) is plural, use 'are' else 'is'
            verb_plur = "are" if _is_plural_surface(subj_list[0]) else "is"
        else:
            subj_phrase = _human_join(subj_list_simple)
            verb_plur = "are"
        if "park" in pred_l:
            rel_sentences.append(f"{subj_phrase} {verb_plur} parked near the {obj_disp}.")
        elif "wear" in pred_l:
            rel_sentences.append(f"{subj_phrase} {verb_plur} wearing noted items.")
        elif "hold" in pred_l or "have" in pred_l:
            rel_sentences.append(f"{subj_phrase} {verb_plur} holding the {obj_disp}.")
        elif pred_l in ("on", "in", "on top of"):
            rel_sentences.append(f"{subj_phrase} {verb_plur} {pred_l} the {obj_disp}.")
        elif pred_l in ("next to", "near", "by", "beside", "along"):
            rel_sentences.append(f"{subj_phrase} {verb_plur} {pred_l} the {obj_disp}.")
        else:
            rel_sentences.append(f"{subj_phrase} {verb_plur} {pred_l} the {obj_disp}.")
    if rel_sentences:
        parts.append(" ".join(rel_sentences[:max_relations]))

    # Attributes: build human-friendly lines but avoid repeats with display
    attr_lines = []
    added = set()
    for oid, display, coarse, sc in facts_objs:
        o = next((x for x in cleaned_objs if int(x["object_id"]) == int(oid)), None)
        if not o:
            continue
        attrs = o.get("attributes") or []
        if not attrs:
            continue
        # produce a human phrase without repeating display words
        s = _normalize_attribute_phrase(coarse, attrs, display)
        # ensure attribute phrase doesn't repeat words in display
        if s:
            # skip if attribute phrase duplicates the display surface
            disp_low = display.lower()
            if any(tok.lower() in disp_low for tok in re.findall(r"\w+", s) if len(tok) > 1) and display.lower() in s.lower():
                # skip duplicate style "orange parking meter that is orange"
                continue
            if s not in added:
                attr_lines.append(s)
                added.add(s)
        if len(attr_lines) >= 4:
            break
    if attr_lines:
        parts.append("Notable details include: " + "; ".join(attr_lines) + ".")

    # Dynamics & conclusion
    if counts.get("person", 0) > 0:
        parts.append("Pedestrians appear to be standing or moving along the sidewalk.")
    if counts.get("vehicle", 0) > 0 or counts.get("bicycle", 0) > 0:
        parts.append("Some vehicles and bikes are parked while others occupy the road.")
    parts.append("Overall, the scene reads as a typical urban street scene.")

    paragraph = " ".join([p for p in parts if p])
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    paragraph = paragraph.replace(" a a ", " a ")
    paragraph = paragraph.replace("on top of the street", "on the street")
    paragraph = paragraph.replace("parked on the street", "parked along the curb")
    return paragraph

# ----------------------
# Simple refinement (1-round)
# ----------------------
def refine_description(entry: Dict, previous_description: str, user_instruction: str) -> str:
    inst = (user_instruction or "").lower()
    base = generate_description_from_entry(entry)
    if not inst:
        return base
    if "pedestrian" in inst or "people" in inst:
        cleaned, facts_objs, _, _, _, _ = _serialize_cleaned_facts(entry, top_k_rels=12)
        ppl = [t for t in facts_objs if t[2]=="person"]
        if ppl:
            rep = ppl[0][1]
            return f"Focus: There are {len(ppl)} pedestrians, including {rep}. {base}"
        return base
    if "vehicle" in inst or "traffic" in inst:
        cleaned, facts_objs, _, _, _, _ = _serialize_cleaned_facts(entry, top_k_rels=12)
        veh = sum(1 for t in facts_objs if t[2]=="vehicle")
        return f"Note: There {'is' if veh==1 else 'are'} {veh} vehicle{'s' if veh!=1 else ''} present. {base}"
    return base + (" Note: " + user_instruction if user_instruction else "")

# ----------------------
# Exposed utilities
# ----------------------
def get_cleaned_entry(entry: Dict) -> Dict:
    return _dedupe_objects(_merge_parts_and_windows(entry))

def get_runtime_info(entry: Optional[Dict] = None) -> Dict:
    info = {"torch": False, "device": "cpu", "salience": "hybrid"}
    if entry is not None:
        try:
            _, _, _, _, _, method = _serialize_cleaned_facts(entry, top_k_rels=6)
            info["salience"] = method
        except Exception:
            info["salience"] = "unknown"
    return info

# CLI demo
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        with open(sys.argv[1]) as fh:
            data = json.load(fh)
        entry = data if isinstance(data, dict) else (data[0] if isinstance(data, list) else data)
    else:
        entry = {
            "image_id": 1,
            "objects": [
                {"object_id": 1, "names":["person"], "attributes":["wearing red jacket"], "x":100,"y":220,"w":50,"h":150},
                {"object_id": 2, "names":["bicycle"], "attributes":["blue","parked"], "x":160,"y":260,"w":80,"h":40},
                {"object_id": 3, "names":["building"], "attributes":["brick"], "x":10,"y":40,"w":400,"h":600},
                {"object_id": 4, "names":["car"], "attributes":["parked","white"], "x":300,"y":280,"w":140,"h":60}
            ],
            "relationships":[{"subject_id":1,"predicate":"riding","object_id":2},{"subject_id":4,"predicate":"parked next to","object_id":3}],
            "scene_attributes":{"weather":"sunny","time_of_day":"afternoon"}
        }
    print("Runtime:", get_runtime_info(entry))
    print("\nCleaned entry keys:", list(get_cleaned_entry(entry).keys()))
    print("\nGenerated description:\n")
    print(generate_description_from_entry(entry))
