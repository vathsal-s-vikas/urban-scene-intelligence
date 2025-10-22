# test_batch.py
import os, json
from PIL import Image
from stage1_object_detection import run_object_detection
from stage2_semantic_scene_graph import build_scene_graph
from description_generator import generate_description_from_entry, refine_description
from app import scene_graph_to_vg_entry  # if you implemented converter there

os.makedirs("data/test_outputs", exist_ok=True)
img_dir = "data/sample_images"
for fname in sorted(os.listdir(img_dir)):
    if not fname.lower().endswith((".jpg",".png","jpeg")):
        continue
    path = os.path.join(img_dir, fname)
    print("Processing", fname)
    detections = run_object_detection(path, save_output=False)
    graph = build_scene_graph(detections, weather="unknown", time_of_day="unknown")
    entry = scene_graph_to_vg_entry(graph)
    desc = generate_description_from_entry(entry)
    out = {
        "image": fname,
        "detections": detections,
        "graph": graph,
        "description": desc
    }
    with open(os.path.join("data/test_outputs", fname + ".json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("Saved", fname + ".json")
