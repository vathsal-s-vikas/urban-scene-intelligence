# app.py

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
import os

from stage1_object_detection import run_object_detection
from stage2_semantic_scene_graph import build_scene_graph, save_scene_graph
from description_generator import generate_narrative_description

st.set_page_config(page_title="Urban Scene Intelligence", layout="wide")
st.title("🏙️ Urban Scene Intelligence — Static Urban Scene Understanding")

st.sidebar.header("Upload Urban Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

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
        graph = build_scene_graph(detections, weather="sunny", time_of_day="daytime")
        save_scene_graph(graph)  # optional: save for inspection
    st.success("Scene graph generated!")

    # ---- Stage 3 ----
    st.subheader("3️⃣ Initial Narrative Description")
    with st.spinner("Generating description..."):
        initial_desc = generate_narrative_description(graph)
    narrative_area = st.text_area("📝 Description", value=initial_desc, height=200)

    # ---- Stage 4: One-round Clarification ----
    st.subheader("4️⃣ Optional Description Refinement")
    clarification = st.text_input(
        "Refine the description (e.g., focus on traffic or pedestrians):",
        value=""
    )

    if st.button("Refine Description") and clarification.strip():
        with st.spinner("Refining description..."):
            # Append user clarification to scene facts
            sentences = [
                f"{n['label']} {' '.join(n.get('attributes',[]))}".strip()
                for n in graph['nodes']
            ]
            # Include edges as sentences
            for e in graph['edges']:
                src = next(n for n in graph['nodes'] if n['id']==e['source'])
                tgt = next(n for n in graph['nodes'] if n['id']==e['target'])
                subj = f"{src['label']} {' '.join(src.get('attributes',[]))}".strip()
                obj = f"{tgt['label']} {' '.join(tgt.get('attributes',[]))}".strip()
                sentences.append(f"{subj} {e['relation']} {obj}")

            # Add user prompt
            sentences.append(f"Focus on: {clarification.strip()}")

            # Build a temporary graph-like dict for T5 input
            temp_graph = {
                "nodes": graph['nodes'],
                "edges": graph['edges'],
                "scene_attributes": graph.get('scene_attributes', {})
            }

            refined_desc = generate_narrative_description(temp_graph)
            narrative_area = refined_desc
            st.text_area("📝 Refined Description", value=refined_desc, height=200)

else:
    st.info("⬆️ Upload an urban street image to start.")
