# stage2_scene_graph_generator.py
import json
import numpy as np
from itertools import combinations

def infer_spatial_relation(bbox1, bbox2):
    """Infer basic geometric relation from bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    center1 = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
    center2 = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)

    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]

    if abs(dx) > abs(dy):
        return "left of" if dx > 0 else "right of"
    else:
        return "above" if dy < 0 else "below"

def build_scene_graph(detections, weather="clear", time_of_day="day"):
    """
    detections: list of dicts, each containing
        { 'label': str, 'bbox': [x1, y1, x2, y2], 'confidence': float }
    """

    nodes, edges = [], []

    # Build nodes
    for i, det in enumerate(detections):
        node = {
            "id": f"n{i+1}",
            "label": det["label"],
            "bbox": det["bbox"],
            "attributes": []   # extend later with color, size, etc.
        }
        nodes.append(node)

    # Build edges (spatial relations)
    for n1, n2 in combinations(nodes, 2):
        rel = infer_spatial_relation(n1["bbox"], n2["bbox"])
        edges.append({"source": n1["id"], "target": n2["id"], "relation": rel})

    # Global scene attributes
    scene_attributes = {
        "scene_type": "urban street",
        "weather": weather,
        "time_of_day": time_of_day
    }

    scene_graph = {"nodes": nodes, "edges": edges, "scene_attributes": scene_attributes}
    return scene_graph

def save_scene_graph(scene_graph, output_path="data/scene_graph.json"):
    with open(output_path, "w") as f:
        json.dump(scene_graph, f, indent=4)
    print(f"✅ Scene graph saved to {output_path}")

if __name__ == "__main__":
    # Example test
    detections = [
        {"label": "car", "bbox": [100, 300, 220, 400], "confidence": 0.89},
        {"label": "building", "bbox": [20, 100, 280, 600], "confidence": 0.95},
        {"label": "tree", "bbox": [300, 250, 370, 420], "confidence": 0.88}
    ]

    graph = build_scene_graph(detections, weather="sunny", time_of_day="daytime")
    save_scene_graph(graph)
