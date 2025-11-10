"""
description_generator.py - Production-Ready Urban Scene Description Generator

Features:
- Aggressive noise filtering for Visual Genome data
- Object deduplication and merging
- Salience-based ranking (area + centrality)
- Relationship clustering and natural language generation
- Scene-level context understanding
- Extensible architecture
"""

from typing import Dict, List, Tuple, Optional, Set
import re
import math
from collections import defaultdict

import random

import random
from collections import defaultdict
import re
class Config:
    # Name filtering
    MIN_NAME_LENGTH = 2
    MAX_NAME_LENGTH = 30

    # Geometry filtering
    MIN_OBJECT_AREA = 100        # minimum bbox area to consider
    IOU_THRESHOLD = 0.5          # for object deduplication

    # Description limits
    MAX_SALIENT_OBJECTS = 10     # max objects to mention
    MAX_RELATIONS = 20           # max relationships to include
    MAX_LAYOUT_ITEMS = 8         # max objects for layout description

    # Salience weights
    AREA_WEIGHT = 0.6            # importance of area
    CENTRALITY_WEIGHT = 0.4      # importance of centrality



# ============================================================
# Text enrichment module
# ============================================================
class TextEnricher:
    """Enhance object mentions and sentence flow."""
    
    ADJECTIVES = {
        "car": ["sleek", "red", "shiny", "rusty", "parked"],
        "person": ["walking", "hurrying", "casually strolling"],
        "child": ["playful", "curious", "small"],
        "tree": ["lush", "tall", "leafy"],
        "building": ["towering", "modern", "brick", "glass-fronted"],
        "bicycle": ["old", "rusty", "blue", "parked"],
        "bus": ["yellow", "parked", "stationed"],
        "streetlight": ["tall", "illuminated", "flickering"],
        "bench": ["wooden", "weathered", "empty"],
        "suv": ["parked", "black", "large"],
        "truck": ["parked", "heavy", "large"],
    }
    
    CONNECTORS = ["while", "as", "nearby", "alongside", "amidst", "beside"]
    
    @staticmethod
    def enrich_object(name: str, attributes: list = None) -> str:
        """Return a rich phrase for an object with optional attributes."""
        adj_list = TextEnricher.ADJECTIVES.get(name, [])
        
        # Prefer object attributes if given
        attr_candidates = [a for a in (attributes or []) if a.lower() not in name.lower()]
        if attr_candidates:
            adj_list = attr_candidates + adj_list
        
        adj_phrase = random.choice(adj_list) if adj_list else ""
        if adj_phrase:
            return f"{TextUtils.article_for(adj_phrase)} {adj_phrase} {name}"
        return f"{TextUtils.article_for(name)} {name}"
    
    @staticmethod
    def join_with_connector(sentences: list) -> str:
        """Randomly connect sentences for smoother flow."""
        if not sentences:
            return ""
        text = sentences[0]
        for s in sentences[1:]:
            connector = random.choice(TextEnricher.CONNECTORS)
            text += f", {connector} {s[0].lower() + s[1:]}"  # lowercase first char
        return text

# ============================================================
# Refactored description generator
# ============================================================
def generate_rich_description(entry):
    """
    Generate a polished natural-language description of the scene.
    Fixes grammar, connector logic, and attribute misuse.
    """
    if not entry:
        return ""

    objects = entry.get("objects", [])
    rels = entry.get("relationships", [])
    scene_attr = entry.get("scene_attributes", {}) or {}

    # 1. Scene summary
    obj_counts = {}
    for o in objects:
        n = (o.get("names") or ["object"])[0].lower()
        base = n if not n.endswith("s") else n[:-1]
        obj_counts[base] = obj_counts.get(base, 0) + 1

    parts = ["This is an urban street scene."]
    if obj_counts:
        summary = ", ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in sorted(obj_counts.items(), key=lambda x: -x[1]))
        parts.append(f"The scene contains {summary}.")

    # 2. Describe visible elements
    vis_elems = []
    for o in objects:
        name = (o.get("names") or ["object"])[0]
        attrs = [a for a in (o.get("attributes") or []) if len(a.split()) <= 3]
        # Filter out nonsense attributes
        attrs = [a for a in attrs if not any(bad in a for bad in ["back", "headlight", "shade", "trunk"])]
        desc = " ".join(attrs[:2] + [name])
        if desc and desc not in vis_elems:
            vis_elems.append(desc)
    if vis_elems:
        parts.append("Visible elements include " + ", ".join(vis_elems[:10]) + ".")

    # 3. Describe relations with context-based grammar
    rel_sentences = []
    for r in rels:
        sid, oid, pred = int(r.get("subject_id")), int(r.get("object_id")), r.get("predicate", "")
        s = next((x for x in objects if int(x.get("object_id")) == sid), None)
        o = next((x for x in objects if int(x.get("object_id")) == oid), None)
        if not s or not o:
            continue
        s_name = (s.get("names") or ["object"])[0]
        o_name = (o.get("names") or ["object"])[0]
        pred = pred.lower().strip()

        # Fix bad predicates
        if pred in ["wearing", "has", "holding"] and s_name not in ["person", "man", "woman", "child"]:
            pred = "has"
        if pred in ["nearby", "beside", "next to"]:
            sent = f"The {s_name} is near the {o_name}."
        elif pred in ["on", "above", "under", "below"]:
            sent = f"The {s_name} is {pred} the {o_name}."
        elif pred in ["parked next to", "parked near"]:
            sent = f"The {s_name} is parked near the {o_name}."
        elif pred in ["wearing"]:
            sent = f"The {s_name} is wearing a {o_name}."
        elif pred in ["with", "containing"]:
            sent = f"The {s_name} has {o_name}."
        else:
            sent = f"The {s_name} is {pred} the {o_name}."
        if sent not in rel_sentences:
            rel_sentences.append(sent)

    if rel_sentences:
        parts.append(" ".join(rel_sentences[:6]))

    # 4. Scene-level context
    ctx = []
    if "weather" in scene_attr:
        ctx.append(f"The weather is {scene_attr['weather']}.")
    if "time_of_day" in scene_attr:
        ctx.append(f"It appears to be {scene_attr['time_of_day']}.")
    if ctx:
        parts.append(" ".join(ctx))

    # 5. Add general concluding line
    parts.append("Pedestrians and vehicles are visible in the scene.")
    parts.append("The scene depicts a typical urban environment.")

    # 6. Clean up
    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("..", ".").replace(".,", ",")
    return text.strip()
import random, re

def generate_storytelling_description(entry):
    """Generate a natural, comprehensive description for a scene."""
    if not entry:
        return ""

    objects = entry.get("objects", [])
    rels = entry.get("relationships", [])
    scene_attr = entry.get("scene_attributes", {}) or {}

    # Filter out noise attributes
    NOISE_ATTRS = {"headlight", "trunk", "shade", "back", "front", "side", "top", "bottom", 
                   "left", "right", "part", "section", "area", "piece"}
    BODY_PARTS = {"shirt", "shoes", "pants", "chin", "hand", "arm", "leg", "face", "head", "eye"}
    
    def _has_name(obj, keywords):
        names = obj.get("names", [])
        if isinstance(names, str):
            names = [names]
        names = [n.lower() for n in names]
        return any(any(k in n for k in keywords) for n in names)
    
    def _get_attributes(obj):
        """Extract meaningful attributes, filtering noise."""
        attrs = obj.get("attributes", [])
        if not attrs:
            return []
        clean = []
        for a in attrs:
            if not isinstance(a, str):
                continue
            a = a.strip().lower()
            if a in NOISE_ATTRS:
                continue
            if len(a) > 2 and len(a.split()) <= 2:
                clean.append(a)
        return clean[:3]

    # Categorize objects
    people = [o for o in objects if _has_name(o, ["person", "man", "woman", "child", "pedestrian"])]
    vehicles = [o for o in objects if _has_name(o, ["car", "truck", "bus", "bike", "bicycle", "vehicle", "suv", "van", "motorcycle"])]
    buildings = [o for o in objects if _has_name(o, ["building", "house", "shop", "store", "structure"])]
    trees = [o for o in objects if _has_name(o, ["tree", "plant", "bush", "foliage"])]
    signs = [o for o in objects if _has_name(o, ["sign", "signpost", "banner"])]
    lights = [o for o in objects if _has_name(o, ["light", "lamp", "streetlight", "traffic light"])]
    street_infra = [o for o in objects if _has_name(o, ["pole", "bench", "sidewalk", "crosswalk", "pavement", "curb"])]
    sky_elements = [o for o in objects if _has_name(o, ["sky", "cloud"])]
    windows = [o for o in objects if _has_name(o, ["window", "door"])]
    
    description = []

    # --- Scene Setting ---
    tod = scene_attr.get("time_of_day", "").lower()
    weather = scene_attr.get("weather", "").lower()
    
    if tod in ["daytime", "afternoon"]:
        if "sunny" in weather or "clear" in weather:
            description.append("The scene shows a city street on a sunny afternoon.")
        elif "cloudy" in weather:
            description.append("The scene shows a city street under cloudy skies.")
        else:
            description.append("The scene shows a city street during daytime.")
    elif tod in ["evening", "dusk"]:
        description.append("The scene shows a city street in the evening.")
    else:
        description.append("The scene shows an urban street.")

    # --- Buildings ---
    if buildings:
        building_attrs = []
        for b in buildings[:3]:
            attrs = _get_attributes(b)
            building_attrs.extend(attrs)
        
        if len(buildings) == 1:
            if building_attrs:
                description.append(f"There is a {building_attrs[0]} building along the street.")
            else:
                description.append("There is a building along the street.")
        else:
            if building_attrs:
                unique_attrs = list(set(building_attrs[:2]))
                description.append(f"There are {len(buildings)} buildings along the street, including {', '.join(unique_attrs)} structures.")
            else:
                description.append(f"There are {len(buildings)} buildings along the street.")
    
    # --- Windows ---
    if windows:
        if len(windows) == 1:
            description.append("A window is visible.")
        else:
            description.append(f"There are multiple windows visible.")

    # --- Sky ---
    if sky_elements:
        sky_attrs = []
        for s in sky_elements[:2]:
            attrs = _get_attributes(s)
            sky_attrs.extend(attrs)
        if sky_attrs:
            description.append(f"The sky appears {sky_attrs[0]}.")
        else:
            description.append("The sky is visible in the background.")

    # --- Vehicles ---
    if vehicles:
        vehicle_inventory = {}
        for v in vehicles:
            v_name = (v.get("names") or ["vehicle"])[0].lower()
            attrs = _get_attributes(v)
            color = next((a for a in attrs if a in ["red", "blue", "black", "white", "silver", "grey", "gray", "yellow", "green", "brown"]), None)
            state = next((a for a in attrs if a in ["parked", "moving", "stopped"]), None)
            
            key = (v_name, color)
            if key not in vehicle_inventory:
                vehicle_inventory[key] = {"count": 1, "state": state, "attrs": attrs}
            else:
                vehicle_inventory[key]["count"] += 1
        
        vehicle_phrases = []
        for (v_name, color), data in list(vehicle_inventory.items())[:5]:
            count = data["count"]
            state = data["state"]
            
            if count > 1:
                desc = f"{count} {v_name}s"
            else:
                desc = f"a {v_name}"
            
            if color:
                if count > 1:
                    desc = f"{count} {color} {v_name}s"
                else:
                    desc = f"a {color} {v_name}"
            
            vehicle_phrases.append(desc)
        
        if len(vehicle_phrases) == 1:
            description.append(f"There is {vehicle_phrases[0]} on the street.")
        elif len(vehicle_phrases) == 2:
            description.append(f"There are two vehicles: {vehicle_phrases[0]} and {vehicle_phrases[1]}.")
        else:
            description.append(f"There are several vehicles including {', '.join(vehicle_phrases[:3])}.")

    # --- Trees ---
    if trees:
        tree_attrs = []
        for t in trees[:3]:
            attrs = _get_attributes(t)
            tree_attrs.extend(attrs)
        
        if len(trees) == 1:
            if tree_attrs:
                description.append(f"There is a {tree_attrs[0]} tree.")
            else:
                description.append("There is a tree.")
        else:
            if tree_attrs:
                unique_attrs = list(set(tree_attrs[:2]))
                description.append(f"There are {len(trees)} trees including {', '.join(unique_attrs)} types.")
            else:
                description.append(f"There are {len(trees)} trees.")

    # --- Signs ---
    if signs:
        if len(signs) == 1:
            description.append("There is a sign.")
        else:
            description.append(f"There are several signs.")

    # --- Lights ---
    if lights:
        if len(lights) == 1:
            description.append("There is a streetlight.")
        else:
            description.append(f"There are multiple lights.")

    # --- Street Infrastructure ---
    if street_infra:
        infra_names = []
        for item in street_infra[:4]:
            name = (item.get("names") or ["element"])[0].lower()
            if name not in ["sidewalk", "street", "road"]:
                infra_names.append(name)
        
        if infra_names:
            unique_infra = list(set(infra_names))
            if len(unique_infra) == 1:
                description.append(f"There is a {unique_infra[0]}.")
            else:
                description.append(f"There is street infrastructure including {', '.join(unique_infra[:3])}.")

    # --- People ---
    if people:
        people_narratives = []
        seen_people = set()
        
        for p in people[:4]:
            p_id = int(p.get("object_id"))
            p_name = (p.get("names") or ["person"])[0].lower()
            
            if p_name in seen_people:
                continue
            seen_people.add(p_name)
            
            # Find relationships
            p_rels = [r for r in rels if int(r.get("subject_id")) == p_id]
            
            added = False
            for rel in p_rels[:2]:
                pred = rel.get("predicate", "").lower()
                o_id = int(rel.get("object_id"))
                o = next((x for x in objects if int(x.get("object_id")) == o_id), None)
                if not o:
                    continue
                    
                o_name = (o.get("names") or ["object"])[0].lower()
                
                if o_name in BODY_PARTS:
                    continue
                
                if "near" in pred or "beside" in pred or "by" in pred or "next to" in pred:
                    people_narratives.append(f"a {p_name} near a {o_name}")
                    added = True
                    break
                elif "hold" in pred or "carrying" in pred:
                    people_narratives.append(f"a {p_name} holding a {o_name}")
                    added = True
                    break
                elif "walk" in pred or "standing" in pred:
                    people_narratives.append(f"a {p_name} {pred.split()[0]}ing")
                    added = True
                    break
            
            if not added:
                attrs = _get_attributes(p)
                if attrs and attrs[0] not in BODY_PARTS:
                    people_narratives.append(f"a {attrs[0]} {p_name}")
                else:
                    people_narratives.append(f"a {p_name}")
        
        if people_narratives:
            if len(people_narratives) == 1:
                description.append(f"There is {people_narratives[0]}.")
            elif len(people_narratives) == 2:
                description.append(f"There are two people: {people_narratives[0]} and {people_narratives[1]}.")
            else:
                description.append(f"There are several people including {', '.join(people_narratives[:3])}.")

    # --- Spatial Relationships ---
    rel_narratives = []
    processed = set()
    
    for r in rels[:4]:
        sid, oid = int(r.get("subject_id")), int(r.get("object_id"))
        pred = r.get("predicate", "").lower()
        
        pair = tuple(sorted([sid, oid]))
        if pair in processed:
            continue
        
        s = next((x for x in objects if int(x.get("object_id")) == sid), None)
        o = next((x for x in objects if int(x.get("object_id")) == oid), None)
        if not s or not o:
            continue
        
        s_name = (s.get("names") or ["object"])[0].lower()
        o_name = (o.get("names") or ["object"])[0].lower()
        
        if s_name in ["person", "man", "woman", "child"] or o_name in ["person", "man", "woman", "child"]:
            continue
        if s_name in BODY_PARTS or o_name in BODY_PARTS:
            continue
        
        if "on" in pred or "atop" in pred:
            rel_narratives.append(f"the {s_name} is on the {o_name}")
        elif "near" in pred or "beside" in pred:
            rel_narratives.append(f"the {s_name} is near the {o_name}")
        elif "attached" in pred or "mounted" in pred:
            rel_narratives.append(f"the {s_name} is attached to the {o_name}")
        
        processed.add(pair)
        if len(rel_narratives) >= 2:
            break
    
    if rel_narratives:
        description.append(f"Also, {', '.join(rel_narratives[:2])}.")

    # --- Simple closing ---
    if description:
        description.append("This completes the scene description.")
    else:
        description.append("No significant objects were detected in the scene.")

    text = " ".join(description)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()
# ============================================================
# CANONICALIZATION & CLEANING
# ============================================================
class NameCleaner:
    """Handles all name cleaning and canonicalization."""
    
    SYNONYM_MAP = {
        "male": "man", "female": "woman", "guy": "man", "lady": "woman",
        "boy": "child", "girl": "child",
        "auto": "car", "automobile": "car",
        "bike": "bicycle", "cycle": "bicycle",
        "roadway": "street", "footpath": "sidewalk",
        "lamp": "streetlight", "lamp post": "streetlight",
        "photograph": "scene", "photo": "scene", "image": "scene",
        "edifice": "building", "structure": "building",
        "vehicle": "car", "taxi cab": "taxi", "sport utility": "suv",
    }
    
    NOISE_PATTERNS = [
        r'^(see|there\s+(are?|is)|shows?|has|displays?)\s+',
        r'\s+(on|in|of|at|shown|noted|found|visible|seen)\.?$',
        r'^(photo|image|scene)\s+',
        r'\s+\.$',  # trailing period
    ]
    
    META_TERMS = {
        "scene", "photo", "image", "view", "display", "shown", "noted",
        "found", "visible", "seen", "nighttime", "daytime", "outdoor",
        "indoor", "outside", "inside", "stories", "story",
    }
    
    @classmethod
    def extract_core_noun(cls, name: str) -> str:
        """Extract meaningful noun from verbose Visual Genome names."""
        if not name:
            return ""
        
        name = name.lower().strip()
        
        # Remove noise patterns
        for pattern in cls.NOISE_PATTERNS:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        # Handle "X on/in Y" -> extract X
        match = re.match(r'^([a-z]+)\s+(?:on|in|at|of)\s+', name, re.IGNORECASE)
        if match:
            name = match.group(1)
        
        # Handle "X shows/has Y" -> take Y
        match = re.match(r'^.+?\s+(?:shows?|has)\s+(.+)', name, re.IGNORECASE)
        if match:
            name = match.group(1)
        
        # Clean whitespace and punctuation
        name = re.sub(r'\s+', ' ', name).strip().rstrip('.,;:')
        
        return name
    
    @classmethod
    def canonicalize(cls, name: str) -> str:
        """Apply synonym mapping."""
        name = name.lower().strip()
        
        # Check for partial matches in synonym map
        for key, value in cls.SYNONYM_MAP.items():
            if key in name:
                return value
        
        return name
    
    @classmethod
    def is_valid_name(cls, name: str) -> bool:
        """Check if name is meaningful for descriptions."""
        if not name or len(name) < Config.MIN_NAME_LENGTH:
            return False
        
        # Check against meta-terms
        name_lower = name.lower()
        if any(term in name_lower for term in cls.META_TERMS):
            return False
        
        # Must contain letters
        if not re.search(r'[a-z]', name, re.IGNORECASE):
            return False
        
        # Skip if ends with punctuation
        if name.endswith('.'):
            return False
        
        return True
    
    @classmethod
    def clean_name(cls, obj: Dict) -> str:
        """Full cleaning pipeline for an object."""
        names = obj.get("names") or []
        
        if names and len(names) > 0:
            name = names[0].replace("_", " ").strip()
            name = cls.extract_core_noun(name)
        else:
            syns = obj.get("synsets") or []
            if syns and len(syns) > 0:
                name = syns[0].split(".")[0].replace("_", " ").strip()
            else:
                name = "object"
        
        name = cls.canonicalize(name)
        return name if cls.is_valid_name(name) else ""


# ============================================================
# SEMANTIC CATEGORIES
# ============================================================
class SemanticCategories:
    """Coarse semantic grouping for urban scene elements."""
    
    CATEGORIES = {
        "person": {"man", "woman", "person", "child", "boy", "girl", "people", 
                   "pedestrian", "cyclist"},
        "vehicle": {"car", "bus", "truck", "van", "bicycle", "motorcycle", 
                    "taxi", "suv", "vehicle", "scooter"},
        "tree": {"tree", "plant", "bush", "vegetation"},
        "infrastructure": {"road", "street", "sidewalk", "building", "pole", 
                          "sign", "traffic light", "crosswalk", "window", 
                          "door", "wall", "streetlight", "bench"},
        "sky": {"sky", "cloud", "sun"},
    }
    
    @classmethod
    def categorize(cls, name: str) -> str:
        """Return coarse category for a name."""
        name_lower = name.lower()
        
        for category, terms in cls.CATEGORIES.items():
            if name_lower in terms:
                return category
        
        return "object"


# ============================================================
# GEOMETRIC UTILITIES
# ============================================================
class GeometryUtils:
    """Bounding box operations."""
    
    @staticmethod
    def bbox_area(obj: Dict) -> float:
        """Calculate bbox area."""
        w = float(obj.get("w", 0))
        h = float(obj.get("h", 0))
        return w * h
    
    @staticmethod
    def bbox_center(obj: Dict) -> Tuple[float, float]:
        """Get bbox center coordinates."""
        x = float(obj.get("x", 0))
        y = float(obj.get("y", 0))
        w = float(obj.get("w", 0))
        h = float(obj.get("h", 0))
        return (x + w / 2, y + h / 2)
    
    @staticmethod
    def iou(obj1: Dict, obj2: Dict) -> float:
        """Calculate intersection over union between two bboxes."""
        x1 = float(obj1.get("x", 0))
        y1 = float(obj1.get("y", 0))
        w1 = float(obj1.get("w", 0))
        h1 = float(obj1.get("h", 0))
        
        x2 = float(obj2.get("x", 0))
        y2 = float(obj2.get("y", 0))
        w2 = float(obj2.get("w", 0))
        h2 = float(obj2.get("h", 0))
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def distance(obj1: Dict, obj2: Dict) -> float:
        """Euclidean distance between bbox centers."""
        c1 = GeometryUtils.bbox_center(obj1)
        c2 = GeometryUtils.bbox_center(obj2)
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


# ============================================================
# OBJECT DEDUPLICATION
# ============================================================
class ObjectDeduplicator:
    """Merge duplicate/overlapping objects."""
    
    @staticmethod
    def deduplicate(objects: List[Dict]) -> List[Dict]:
        """Merge highly overlapping objects with similar names."""
        if not objects:
            return []
        
        # Sort by area (largest first)
        sorted_objs = sorted(objects, key=GeometryUtils.bbox_area, reverse=True)
        
        merged = []
        used_indices = set()
        
        for i, obj1 in enumerate(sorted_objs):
            if i in used_indices:
                continue
            
            # Start a cluster with this object
            cluster = [obj1]
            name1 = NameCleaner.clean_name(obj1)
            
            for j, obj2 in enumerate(sorted_objs[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                name2 = NameCleaner.clean_name(obj2)
                
                # Check if they should be merged
                iou = GeometryUtils.iou(obj1, obj2)
                same_name = (name1 == name2)
                
                if iou > Config.IOU_THRESHOLD and same_name:
                    cluster.append(obj2)
                    used_indices.add(j)
            
            # Merge cluster into single object
            merged_obj = ObjectDeduplicator._merge_cluster(cluster)
            merged.append(merged_obj)
            used_indices.add(i)
        
        return merged
    
    @staticmethod
    def _merge_cluster(cluster: List[Dict]) -> Dict:
        """Merge a cluster of objects into one representative."""
        if len(cluster) == 1:
            return cluster[0]
        
        # Take bbox from largest object
        largest = max(cluster, key=GeometryUtils.bbox_area)
        merged = dict(largest)
        
        # Merge attributes (unique)
        all_attrs = set()
        for obj in cluster:
            attrs = obj.get("attributes") or []
            all_attrs.update(a.strip() for a in attrs if isinstance(a, str))
        
        merged["attributes"] = sorted(list(all_attrs))
        
        return merged


# ============================================================
# SALIENCE COMPUTATION
# ============================================================
class SalienceComputer:
    """Compute object importance scores."""
    
    @staticmethod
    def compute(objects: List[Dict], image_width: int = 800, image_height: int = 600) -> Dict[int, float]:
        """
        Compute salience scores combining area and centrality.
        Returns: {object_id: salience_score}
        """
        if not objects:
            return {}
        
        scores = {}
        img_center = (image_width / 2, image_height / 2)
        max_distance = math.sqrt(img_center[0]**2 + img_center[1]**2)
        
        # Calculate max area for normalization
        areas = [GeometryUtils.bbox_area(o) for o in objects]
        max_area = max(areas) if areas else 1.0
        
        for obj in objects:
            oid = int(obj.get("object_id", 0))
            
            # Area component (larger = more salient)
            area = GeometryUtils.bbox_area(obj)
            area_score = area / max_area if max_area > 0 else 0.0
            
            # Centrality component (closer to center = more salient)
            center = GeometryUtils.bbox_center(obj)
            dist = math.sqrt((center[0] - img_center[0])**2 + (center[1] - img_center[1])**2)
            centrality_score = 1.0 - (dist / max_distance) if max_distance > 0 else 0.0
            
            # Combined score
            scores[oid] = (Config.AREA_WEIGHT * area_score + 
                          Config.CENTRALITY_WEIGHT * centrality_score)
        
        return scores


# ============================================================
# RELATIONSHIP PROCESSING
# ============================================================
class RelationshipProcessor:
    """Clean and rank relationships."""
    
    WEAK_PREDICATES = {
        "is", "are", "was", "were", "see", "visible", "appears", 
        "looks", "looking", "shown", "shows", "found", "noted", 
        "taken", "of", "see street", "see tail  light"
    }
    
    @staticmethod
    def collect_relations(entry: Dict, id2name: Dict[int, str]) -> List[Tuple[int, str, int]]:
        """Extract and clean relationship triples."""
        rels = []
        
        for r in entry.get("relationships", []):
            try:
                sid = int(r.get("subject_id"))
                oid = int(r.get("object_id"))
                pred = (r.get("predicate") or "").strip().lower()
                
                if not pred or len(pred) < 2:
                    continue
                
                # Skip weak predicates
                if pred in RelationshipProcessor.WEAK_PREDICATES:
                    continue
                
                # Skip if subject/object not in cleaned objects
                if sid not in id2name or oid not in id2name:
                    continue
                
                # Skip self-references
                subj_name = id2name[sid]
                obj_name = id2name[oid]
                if subj_name == obj_name:
                    continue
                if subj_name in obj_name or obj_name in subj_name:
                    continue
                
                rels.append((sid, pred, oid))
                
            except Exception:
                continue
        
        return rels
    
    @staticmethod
    def normalize_predicate(pred: str) -> str:
        """Normalize predicate to canonical form."""
        pred = pred.lower().strip()
        
        # Spatial relations
        if pred in ("on", "on top of", "atop"):
            return "on"
        if pred in ("by", "near", "next to", "beside", "along"):
            return "near"
        if pred in ("in front of", "in front"):
            return "in front of"
        if pred in ("behind", "in back of"):
            return "behind"
        if "park" in pred:
            return "parked near"
        
        # Attribute relations
        if "wear" in pred or "has" in pred:
            return "wearing"
        
        return pred


# ============================================================
# TEXT GENERATION UTILITIES
# ============================================================
class TextUtils:
    """Natural language generation helpers."""
    
    @staticmethod
    def article_for(word: str) -> str:
        """Get appropriate article (a/an)."""
        if not word:
            return "a"
        return "an" if word[0].lower() in "aeiou" else "a"
    
    @staticmethod
    def pluralize(label: str, count: int) -> str:
        """Create count phrase (e.g., '3 cars')."""
        if count == 1:
            return f"1 {label}"
        
        if label.endswith("y") and label not in ("day", "boy", "toy", "way"):
            plural = label[:-1] + "ies"
        elif label.endswith("s"):
            plural = label
        else:
            plural = label + "s"
        
        return f"{count} {plural}"
    
    @staticmethod
    def human_join(items: List[str]) -> str:
        """Join list with commas and 'and'."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + ", and " + items[-1]


# ============================================================
# MAIN GENERATOR
# ============================================================
def generate_description(
    entry: Dict,
    max_objects: int = None,
    max_relations: int = None,
) -> str:
    """
    Generate natural language description from scene graph.
    
    Args:
        entry: Visual Genome-style scene graph dict
        max_objects: Max salient objects to mention (default: Config.MAX_SALIENT_OBJECTS)
        max_relations: Max relationships to describe (default: Config.MAX_RELATIONS)
    
    Returns:
        Natural language description string
    """
    if max_objects is None:
        max_objects = Config.MAX_SALIENT_OBJECTS
    if max_relations is None:
        max_relations = Config.MAX_RELATIONS
    
    # Step 1: Clean and filter objects
    raw_objects = entry.get("objects", [])
    
    cleaned_objects = []
    for obj in raw_objects:
        name = NameCleaner.clean_name(obj)
        if not name:
            continue
        
        area = GeometryUtils.bbox_area(obj)
        if area < Config.MIN_OBJECT_AREA:
            continue
        
        cleaned_objects.append(obj)
    
    # Step 2: Deduplicate
    cleaned_objects = ObjectDeduplicator.deduplicate(cleaned_objects)
    
    if not cleaned_objects:
        return "This is an urban street scene."
    
    # Step 3: Compute salience
    salience_scores = SalienceComputer.compute(cleaned_objects)
    
    # Sort by salience
    sorted_objects = sorted(
        cleaned_objects,
        key=lambda o: salience_scores.get(int(o.get("object_id", 0)), 0.0),
        reverse=True
    )
    
    # Build id->name mapping
    id2name = {
        int(o.get("object_id")): NameCleaner.clean_name(o)
        for o in sorted_objects
    }
    
    # Step 4: Process relationships
    relations = RelationshipProcessor.collect_relations(entry, id2name)
    
    # Normalize predicates
    normalized_rels = [
        (sid, RelationshipProcessor.normalize_predicate(pred), oid)
        for sid, pred, oid in relations
    ]
    
    # Step 5: Generate text
    parts = []
    
    # Opening
    parts.append("This is an urban street scene.")
    
    # Count summary
    category_counts = defaultdict(int)
    for obj in sorted_objects:
        name = id2name.get(int(obj.get("object_id")), "")
        if name:
            category = SemanticCategories.categorize(name)
            category_counts[category] += 1
    
    if category_counts:
        # Remove generic "object" category from summary
        if "object" in category_counts:
            del category_counts["object"]
        
        sorted_counts = sorted(category_counts.items(), key=lambda x: -x[1])[:4]
        count_phrases = [TextUtils.pluralize(cat, cnt) for cat, cnt in sorted_counts]
        
        if count_phrases:
            parts.append("The scene contains " + TextUtils.human_join(count_phrases) + ".")
    
    # Layout description (top salient objects)
    layout_items = []
    for obj in sorted_objects[:Config.MAX_LAYOUT_ITEMS]:
        name = id2name.get(int(obj.get("object_id")), "")
        if not name:
            continue
        
        attrs = obj.get("attributes") or []
        clean_attrs = [
            a.strip() for a in attrs 
            if isinstance(a, str) and len(a.strip()) > 1 and a.strip().lower() not in name.lower()
        ]
        
        category = SemanticCategories.categorize(name)
        
        # Build descriptive phrase
        if clean_attrs:
            color_attrs = [a for a in clean_attrs if a.lower() in 
                          ["red", "blue", "yellow", "black", "white", "silver", "grey", "gray", "green"]]
            if color_attrs:
                layout_items.append(f"{TextUtils.article_for(color_attrs[0])} {color_attrs[0]} {name}")
            else:
                layout_items.append(f"{TextUtils.article_for(name)} {clean_attrs[0]} {name}")
        else:
            layout_items.append(f"{TextUtils.article_for(name)} {name}")
    
    if layout_items:
        parts.append("Visible elements include " + TextUtils.human_join(layout_items) + ".")
    
    # Relationships
    # Group by (predicate, object) to create natural sentences
    rel_groups = defaultdict(set)
    for sid, pred, oid in normalized_rels[:max_relations * 2]:
        rel_groups[(pred, id2name[oid])].add(id2name[sid])
    
    rel_sentences = []
    for (pred, obj_name), subj_set in list(rel_groups.items())[:max_relations]:
        subj_list = sorted(list(subj_set))[:2]  # Limit subjects
        
        if len(subj_list) == 1:
            subj_phrase = f"The {subj_list[0]}"
            verb = "is"
        else:
            subj_phrase = f"The {subj_list[0]} and {subj_list[1]}"
            verb = "are"
        
        # Build sentence based on predicate
        if pred in ("on", "near", "in", "behind", "in front of"):
            rel_sentences.append(f"{subj_phrase} {verb} {pred} the {obj_name}.")
        elif pred == "wearing":
            rel_sentences.append(f"{subj_phrase} {verb} wearing {obj_name}.")
        elif pred == "parked near":
            rel_sentences.append(f"{subj_phrase} {verb} parked near the {obj_name}.")
        else:
            rel_sentences.append(f"{subj_phrase} {verb} {pred} the {obj_name}.")
    
    if rel_sentences:
        parts.append(" ".join(rel_sentences))
    
    # Context-aware closing
    if category_counts.get("person", 0) > 0:
        parts.append("Pedestrians are visible in the scene.")
    
    if category_counts.get("vehicle", 0) > 1:
        parts.append("Multiple vehicles are present.")
    
    parts.append("The scene depicts a typical urban environment.")
    
    # Final cleanup
    text = " ".join(parts)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    text = re.sub(r'\.+', '.', text)
    
    return text.strip()


# ============================================================
# UTILITY FUNCTIONS FOR INTEGRATION
# ============================================================
def compute_node_salience(entry: Dict) -> Tuple[List[float], List[int], str]:
    """
    Compute salience scores for all objects.
    
    Returns:
        (scores, object_ids, method_name)
    """
    objects = entry.get("objects", [])
    scores_dict = SalienceComputer.compute(objects)
    
    object_ids = [int(o.get("object_id")) for o in objects]
    scores = [scores_dict.get(oid, 0.0) for oid in object_ids]
    
    return scores, object_ids, "area_centrality_combined"


def _serialize_facts(entry: Dict, max_rels: int = 20) -> Tuple:
    """
    Return cleaned entry and serialized facts for debugging.
    
    Returns:
        (cleaned_entry, facts_objects, relations_sorted, scene_desc, id2score, method)
    """
    # Clean objects
    raw_objects = entry.get("objects", [])
    cleaned_objects = []
    
    for obj in raw_objects:
        name = NameCleaner.clean_name(obj)
        if name and GeometryUtils.bbox_area(obj) >= Config.MIN_OBJECT_AREA:
            cleaned_objects.append(obj)
    
    cleaned_objects = ObjectDeduplicator.deduplicate(cleaned_objects)
    
    # Compute salience
    id2score = SalienceComputer.compute(cleaned_objects)
    
    # Build facts list
    facts_objs = []
    for obj in cleaned_objects:
        oid = int(obj.get("object_id"))
        name = NameCleaner.clean_name(obj)
        category = SemanticCategories.categorize(name)
        score = id2score.get(oid, 0.0)
        facts_objs.append((oid, name, category, score))
    
    # Sort by salience
    facts_objs.sort(key=lambda x: x[3], reverse=True)
    
    # Build id2name
    id2name = {int(o.get("object_id")): NameCleaner.clean_name(o) for o in cleaned_objects}
    
    # Collect relations
    relations = RelationshipProcessor.collect_relations(entry, id2name)
    rels_sorted = [(sid, RelationshipProcessor.normalize_predicate(p), oid) 
                   for sid, p, oid in relations[:max_rels]]
    
    # Scene attributes
    scene_attrs = entry.get("scene_attributes", {})
    
    cleaned_entry = {
        "objects": cleaned_objects,
        "relationships": entry.get("relationships", []),
        "scene_attributes": scene_attrs,
        "image_id": entry.get("image_id"),
    }
    
    return (cleaned_entry, facts_objs, rels_sorted, scene_attrs, id2score, "area_centrality_combined")


def get_runtime_info(entry: Dict) -> Dict:
    """Return runtime statistics."""
    objects = entry.get("objects", [])
    relations = entry.get("relationships", [])
    
    cleaned_count = sum(1 for o in objects if NameCleaner.clean_name(o))
    
    return {
        "total_objects": len(objects),
        "cleaned_objects": cleaned_count,
        "total_relations": len(relations),
        "config": {
            "area_weight": Config.AREA_WEIGHT,
            "centrality_weight": Config.CENTRALITY_WEIGHT,
            "iou_threshold": Config.IOU_THRESHOLD,
        }
    }