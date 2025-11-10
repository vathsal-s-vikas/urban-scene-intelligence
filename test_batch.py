# test_batch.py
import os, json
from PIL import Image
from scenegraph import build_scene_graph
from description_generator import generate_description, refine_description, generate_rich_description, generate_storytelling_description

os.makedirs("data/test_outputs", exist_ok=True)
img_dir = "data/sample_images"
for fname in sorted(os.listdir(img_dir)):
    if not fname.lower().endswith((".jpg",".png","jpeg")):
        continue
    path = os.path.join(img_dir, fname)
    print("Processing", fname)
    graph = build_scene_graph()
    desc = generate_description()
    out = {
        "image": fname,
        "graph": graph,
        "description": desc
    }
    with open(os.path.join("data/test_outputs", fname + ".json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("Saved", fname + ".json")
