# app.py (updated - integrates graph-aware description generator)
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
import os
import json

from stage1_object_detection import run_object_detection
from stage2_semantic_scene_graph import build_scene_graph, save_scene_graph
# new graph-aware generator functions
from description_generator import generate_description_from_entry, refine_description

st.set_page_config(page_title="Urban Scene Intelligence", layout="wide")
st.title("🏙️ Urban Scene Intelligence — Static Urban Scene Understanding (Graph-aware)")

st.sidebar.header("Upload Urban Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# ---------------------------
# Helper: convert stage2 graph -> Visual Genome style entry
# ---------------------------
def scene_graph_to_vg_entry(graph: dict) -> dict:
    """
    Convert structured scene graph dict (nodes with 'id','label','bbox','attributes',
    edges with 'source','target','relation') to Visual Genome-like entry:
    {
      'image_id': ...,
      'objects': [ {object_id, names, attributes, x,y,w,h, synsets?}, ... ],
      'relationships': [ {subject_id, predicate, object_id}, ... ],
      'scene_attributes': {...}  # if present
    }
    """
    # nodes: expect list of dicts with 'id' like 'n1' or numeric
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    scene_attrs = graph.get("scene_attributes", {})
    # map node id string -> integer id
    id_map = {}
    objects = []
    for idx, n in enumerate(nodes):
        nid = n.get("id", None)
        # try to extract numeric id if 'n12' else fallback to sequential
        try:
            if isinstance(nid, str) and nid.startswith("n"):
                oid = int(nid[1:])
            elif isinstance(nid, int):
                oid = int(nid)
            else:
                oid = idx + 1
        except Exception:
            oid = idx + 1
        id_map[nid] = oid
        # bbox handling: graph may provide [x1,y1,x2,y2] or something else
        bbox = n.get("bbox", None)
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            x = float(x1)
            y = float(y1)
            w = float(x2) - float(x1)
            h = float(y2) - float(y1)
        else:
            # fallback: if node has 'x','y','w','h' use them, else zeros
            x = float(n.get("x", 0))
            y = float(n.get("y", 0))
            w = float(n.get("w", 0))
            h = float(n.get("h", 0))
        name = n.get("label") or (n.get("names")[0] if n.get("names") else "object")
        attributes = n.get("attributes", []) or []
        obj = {
            "object_id": oid,
            "names": [name] if isinstance(name, str) else name,
            "attributes": attributes,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            # optional: include synsets if you have them
        }
        objects.append(obj)
    # relationships
    relationships = []
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        # source/target in stage2 might be 'car_1' or 'n1'
        # attempt to map using id_map; if not found, try splits
        def map_to_oid(val):
            if val in id_map:
                return id_map[val]
            if isinstance(val, str):
                # try to extract trailing digits
                import re
                m = re.search(r"(\\d+)$", val)
                if m:
                    return int(m.group(1))
            # last resort: attempt int
            try:
                return int(val)
            except Exception:
                return None
        sid = map_to_oid(src)
        oid = map_to_oid(tgt)
        if sid is None or oid is None:
            continue
        pred = e.get("relation") or e.get("predicate") or ""
        relationships.append({"subject_id": sid, "predicate": pred, "object_id": oid})
    entry = {
        "image_id": graph.get("image_id", None) or -1,
        "objects": objects,
        "relationships": relationships,
    }
    if scene_attrs:
        entry["scene_attributes"] = scene_attrs
    return entry

# ---------------------------
# App main
# ---------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Scene", use_container_width=True)

    image_path = os.path.join("data", "uploaded_image.jpg")
    os.makedirs("data", exist_ok=True)
    img.save(image_path)

    # ---- Stage 1 ----
    st.subheader("1️⃣ Object Detection")
    with st.spinner("Detecting objects..."):
        detections = run_object_detection(image_path, save_output=False)
    st.success(f"Detected {len(detections)} objects.")
    st.write(", ".join([d['label'] for d in detections]))

    # ---- Stage 2 ----
    st.subheader("2️⃣ Building Structured Scene Graph")
    with st.spinner("Constructing scene graph..."):
        # build_scene_graph should accept detections (list of dicts as returned by run_object_detection)
        # and return a structured graph dict with 'nodes' and 'edges'
        graph = build_scene_graph(detections, weather=st.sidebar.text_input("Weather", "sunny"),
                                  time_of_day=st.sidebar.text_input("Time of day", "daytime"))
        # optionally save for inspection
        save_scene_graph(graph)
    st.success("Scene graph generated!")
    # show JSON (collapsible)
    with st.expander("Show scene graph JSON"):
        st.json(graph)

    # Convert scene graph to Visual Genome-style entry for generator
    entry = scene_graph_to_vg_entry(graph)

    # ---- Stage 3 ----
    st.subheader("3️⃣ Initial Narrative Description")
    with st.spinner("Generating description..."):
        # new function from description_generator
        initial_desc = generate_description_from_entry(entry)
    st.text_area("📝 Description", value=initial_desc, height=220)

    # ---- Stage 4: One-round Clarification ----
    st.subheader("4️⃣ Optional Description Refinement")
    clarification = st.text_input("Refine the description (e.g., focus on traffic or pedestrians):", value="")

    if st.button("Refine Description") and clarification.strip():
        with st.spinner("Refining description..."):
            refined_desc = refine_description(entry, initial_desc, clarification.strip())
        st.text_area("📝 Refined Description", value=refined_desc, height=220)
else:
    st.info("⬆️ Upload an urban street image to start.")
