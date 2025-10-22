# test_graph_app.py
"""
Streamlit test harness (updated) to surface cleaned/merged facts from description_generator.

Key features:
 - Robust imports from description_generator (handles missing names).
 - If description_generator._serialize_facts exists, display the cleaned entry and facts used for generation.
 - Handles compute_node_salience returning 2- or 3-tuple.
 - Toggle to draw original vs cleaned bounding boxes on matched image.
"""

import streamlit as st
import json, os, re
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Try imports from description_generator (be permissive)
gen_fn = None
refine_fn = None
compute_node_salience = None
_serialize_facts = None
get_runtime_info = None
debug_prompt = None

try:
    from description_generator import generate_description_from_entry
    gen_fn = generate_description_from_entry
except Exception as e:
    st.warning(f"generate_description_from_entry not available: {e}")

try:
    from description_generator import refine_description
    refine_fn = refine_description
except Exception:
    refine_fn = None

try:
    from description_generator import compute_node_salience
except Exception:
    compute_node_salience = None

# Prefer _serialize_facts name used in the updated generator; fall back to older names
try:
    from description_generator import _serialize_facts
except Exception:
    try:
        from description_generator import _serialize_facts_with_salience as _serialize_facts
    except Exception:
        _serialize_facts = None

try:
    from description_generator import get_runtime_info
except Exception:
    get_runtime_info = None

try:
    from description_generator import debug_prompt
except Exception:
    debug_prompt = None

st.set_page_config(page_title="Scene Graph Test Bench (cleaned facts)", layout="wide")
st.title("🧪 Scene-Graph Test Bench — Cleaned Facts & Description QA")

# -------------------------
# Helpers
# -------------------------
def normalize_token(tok):
    return re.sub(r"[^a-z0-9]", "", tok.lower())

def detect_hallucinations(entry, text):
    present = set()
    for o in entry.get("objects", []):
        names = o.get("names") or []
        if names:
            present.add(normalize_token(names[0]))
        for a in (o.get("attributes") or []):
            present.add(normalize_token(a))
    found = set()
    words = re.findall(r"[A-Za-z0-9]+(?: [A-Za-z0-9]+)?", text)
    for w in words:
        k = normalize_token(w)
        found.add(k)
    URBAN_LEX = {"car","bus","truck","bicycle","motorcycle","bike","taxi","van","person","man","woman","child",
                 "pedestrian","building","tree","trafficlight","bench","crosswalk","road","street","sidewalk","lamp","van","worktruck","parkingmeter"}
    halluc = [w for w in found if (w in URBAN_LEX and w not in present)]
    return sorted(set(halluc))

def draw_bboxes_on_pil(image: Image.Image, objects, color="red"):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    for o in objects:
        x = float(o.get("x", 0)); y = float(o.get("y", 0))
        w = float(o.get("w", 0)); h = float(o.get("h", 0))
        x2 = x + w; y2 = y + h
        draw.rectangle([x, y, x2, y2], outline=color, width=2)
        name = (o.get("names") or ["object"])[0]
        label = f"{name}"
        text_xy = (x, max(0, y-16))
        draw.rectangle([text_xy[0]-1, text_xy[1]-1, text_xy[0]+len(label)*7+6, text_xy[1]+16], fill=color)
        draw.text(text_xy, label, fill="white", font=font)
    return image

# -------------------------
# Sidebar: load JSON / images / folder
# -------------------------
st.sidebar.header("Load test entries & images")
json_file = st.sidebar.file_uploader("Upload VisualGenome-style JSON (list of entries)", type=["json"])
use_sample = st.sidebar.checkbox("Use built-in sample entry", value=True)

st.sidebar.markdown("**Images**")
uploaded_imgs = st.sidebar.file_uploader("Upload image files (optional; multiple)", type=["jpg","jpeg","png"], accept_multiple_files=True)
local_folder = st.sidebar.text_input("Local image folder path (optional)", value="")  # e.g. data/test_images

# save uploaded files to data/test_images for matching
IMG_DIR = "data/test_images"
os.makedirs(IMG_DIR, exist_ok=True)
uploaded_map = {}
if uploaded_imgs:
    for f in uploaded_imgs:
        save_path = os.path.join(IMG_DIR, f.name)
        with open(save_path, "wb") as out:
            out.write(f.getbuffer())
        uploaded_map[f.name] = save_path

# list files from local folder (if exists)
local_map = {}
if local_folder:
    if os.path.isdir(local_folder):
        for fn in os.listdir(local_folder):
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                local_map[fn] = os.path.join(local_folder, fn)
    else:
        st.sidebar.warning("Local folder path not found on server. Leave blank or correct the path.")

# merged image pool
image_pool = dict(local_map)
image_pool.update(uploaded_map)

# -------------------------
# Load entries
# -------------------------
entries = []
if json_file is not None:
    try:
        entries = json.load(json_file)
        if isinstance(entries, dict):
            entries = [entries]
    except Exception as e:
        st.sidebar.error(f"Failed to parse JSON: {e}")
elif use_sample:
    entries = [
        {
            "image_id": 1,
            "image_filename": "1.jpg",
            "objects": [
                {"object_id": 1, "names": ["person"], "attributes": ["wearing red jacket"], "x": 100, "y": 220, "w": 50, "h": 150},
                {"object_id": 2, "names": ["bicycle"], "attributes": ["blue"], "x": 160, "y": 260, "w": 80, "h": 40},
                {"object_id": 3, "names": ["building"], "attributes": ["brick"], "x": 10, "y": 40, "w": 400, "h": 600},
                {"object_id": 4, "names": ["car"], "attributes": ["parked"], "x": 300, "y": 280, "w": 140, "h": 60}
            ],
            "relationships": [
                {"subject_id": 1, "predicate": "riding", "object_id": 2},
                {"subject_id": 4, "predicate": "parked next to", "object_id": 3}
            ],
            "scene_attributes": {"weather": "sunny", "time_of_day": "afternoon"}
        }
    ]
else:
    st.sidebar.info("Upload a JSON file or enable built-in sample.")
    st.stop()

# entry selector
idx = st.sidebar.selectbox("Choose entry index", list(range(len(entries))),
                           format_func=lambda i: f"Entry {i} (image_id={entries[i].get('image_id','N/A')})")
entry = entries[idx]

# JSON editor
st.markdown("## Entry JSON (view / edit)")
with st.expander("View / Edit scene graph JSON (edit then Apply)"):
    entry_json_str = st.text_area("Entry JSON", value=json.dumps(entry, indent=2), height=280, key="entry_json")
    if st.button("Apply JSON edits"):
        try:
            edited = json.loads(entry_json_str)
            entry = edited
            entries[idx] = edited
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

# -------------------------
# match image
# -------------------------
def find_image_for_entry(entry):
    if "image_filename" in entry and entry["image_filename"]:
        fn = os.path.basename(entry["image_filename"])
        if fn in image_pool:
            return image_pool[fn]
    if "image_id" in entry and entry["image_id"] is not None:
        s = str(entry["image_id"])
        for fn, p in image_pool.items():
            if s in fn:
                return p
    objs = entry.get("objects", [])
    if objs:
        fn_match = (objs[0].get("names") or [""])[0].lower()
        for fn, p in image_pool.items():
            if fn_match and fn_match in fn.lower():
                return p
    return None

matched_path = find_image_for_entry(entry)

st.markdown("## Optional Image (matching)")
if matched_path:
    st.success(f"Matched image: {os.path.basename(matched_path)}")
    try:
        pil_img = Image.open(matched_path).convert("RGB")
    except Exception as e:
        st.warning("Failed to open matched image: " + str(e))
        pil_img = None
else:
    uploaded_img = st.file_uploader("Upload related image (optional) — JPEG/PNG", type=["jpg","jpeg","png"], key="img_upload")
    if uploaded_img:
        pil_img = Image.open(uploaded_img).convert("RGB")
    else:
        pil_img = None
        st.info("No matched image found. Provide images or add 'image_filename' to entry.")

# -------------------------
# Show serialized facts (cleaned) if available
# -------------------------
st.markdown("## Cleaned Facts (from description_generator)")
cleaned = None
facts_objs = None
rels_sorted = None
scene_desc = None
id2score = None
salience_method = None

if _serialize_facts:
    try:
        # signature: cleaned, facts_objs, rels_sorted, scene_desc, id2score, method
        cleaned, facts_objs, rels_sorted, scene_desc, id2score, salience_method = _serialize_facts(entry, max_rels=20)
        st.write("Salience method:", salience_method)
        st.write("Cleaned objects (post-merge/dedupe):")
        # present cleaned objects nicely
        for o in cleaned.get("objects", []):
            st.write(f"- id={o.get('object_id')} name={ (o.get('names') or [''])[0] } attrs={o.get('attributes') or []} bbox={(o.get('x'),o.get('y'),o.get('w'),o.get('h'))}")
        st.write("Serialized facts (salience-ordered):")
        # facts_objs is list of tuples (oid, display, coarse, score) in our generator
        if isinstance(facts_objs, list) and facts_objs and isinstance(facts_objs[0], (list,tuple)):
            st.write([{"object_id": int(t[0]), "display": t[1], "coarse": t[2], "score": float(t[3])} for t in facts_objs])
        else:
            st.write(facts_objs)
        if rels_sorted:
            st.write("Top relations (triples):")
            # convert to human strings
            rtxts = []
            for sid, pred, oid in rels_sorted:
                s = next((x for x in cleaned.get("objects", []) if int(x.get("object_id"))==int(sid)), None)
                o = next((x for x in cleaned.get("objects", []) if int(x.get("object_id"))==int(oid)), None)
                if s and o:
                    rtxts.append(f"{(s.get('names') or [''])[0]} {pred} {(o.get('names') or [''])[0]}")
                else:
                    rtxts.append(f"{sid} {pred} {oid}")
            st.write(rtxts)
        if scene_desc:
            st.write("Scene attributes:", scene_desc)
    except Exception as e:
        st.write("Error calling _serialize_facts:", e)
        cleaned = None
else:
    st.info("description_generator._serialize_facts not available — using local heuristics.")
    # show local salient objects & relations
    sorted_objs = sorted(entry.get("objects", []), key=lambda o: float(o.get("w",0))*float(o.get("h",0)), reverse=True)
    salient = []
    for o in sorted_objs[:8]:
        name = (o.get("names") or ["object"])[0]
        attrs = o.get("attributes") or []
        if attrs:
            salient.append(f"{', '.join(attrs[:2])} {name}")
        else:
            salient.append(name)
    st.write("Salient objects (area heuristic):", salient)
    rels_local = []
    for r in entry.get("relationships", []):
        sid = int(r.get("subject_id")); oid = int(r.get("object_id"))
        s = next((x for x in entry.get("objects",[]) if int(x.get("object_id"))==sid), None)
        o = next((x for x in entry.get("objects",[]) if int(x.get("object_id"))==oid), None)
        if s and o:
            rels_local.append(f"{(s.get('names') or ['object'])[0]} {r.get('predicate','')} {(o.get('names') or ['object'])[0]}")
    st.write("Top relation facts:", rels_local[:8])

# -------------------------
# Show matched image with original and cleaned boxes (toggle)
# -------------------------
st.markdown("## Visualization")
show_cleaned_bboxes = st.checkbox("Show cleaned object bounding boxes (if available)", value=True)
if pil_img:
    vis = pil_img.copy()
    if show_cleaned_bboxes and cleaned:
        vis = draw_bboxes_on_pil(vis, cleaned.get("objects", []), color="lime")
        st.image(vis, caption="Image with CLEANED bboxes (lime)", use_column_width=True)
    else:
        vis = draw_bboxes_on_pil(vis, entry.get("objects", []), color="red")
        st.image(vis, caption="Image with ORIGINAL bboxes (red)", use_column_width=True)
else:
    st.info("No image to visualize. Upload one in sidebar to see bboxes.")

# -------------------------
# Show compute_node_salience if available
# -------------------------
st.markdown("## Salience (compute_node_salience)")
if compute_node_salience:
    try:
        sal_ret = compute_node_salience(entry)
        if isinstance(sal_ret, tuple):
            if len(sal_ret) == 3:
                scores, obj_ids, method = sal_ret
            elif len(sal_ret) == 2:
                scores, obj_ids = sal_ret
                method = "unknown"
            else:
                scores, obj_ids = sal_ret[0], sal_ret[1]
                method = sal_ret[2] if len(sal_ret) > 2 else "unknown"
            st.write("Method:", method)
            # pair them
            st.json({str(int(obj_ids[i])): float(scores[i]) for i in range(min(len(obj_ids), len(scores)))})
        else:
            st.write("Unexpected compute_node_salience return:", sal_ret)
    except Exception as e:
        st.write("Error calling compute_node_salience:", e)
else:
    st.write("compute_node_salience not available in description_generator (area fallback used).")

# runtime info if provided by generator
if get_runtime_info:
    try:
        runtime = get_runtime_info(entry)
        st.write("Runtime info:", runtime)
    except Exception as e:
        st.write("get_runtime_info error:", e)

# -------------------------
# Generation controls
# -------------------------
st.markdown("## Generation")
gen_col1, gen_col2 = st.columns([1,2])
with gen_col1:
    run_gen = st.button("Generate description")
    run_refine = st.button("Refine description (using instruction)")
with gen_col2:
    user_instruction = st.text_input("Refinement instruction (one-round)", value="Focus on pedestrians and their actions.")
    show_halluc = st.checkbox("Highlight detected hallucinations (heuristic)", value=True)
    show_debug_prompt = st.checkbox("Show debug prompt (if available)", value=False)

if run_gen:
    if gen_fn is None:
        st.error("generate_description_from_entry not available. Ensure description_generator exports it.")
    else:
        with st.spinner("Generating..."):
            try:
                desc = gen_fn(entry)
            except Exception as e:
                st.error("Generation error: " + str(e)); desc = ""
        st.markdown("### Generated Description")
        st.write(desc or "(empty)")
        if show_halluc:
            flags = detect_hallucinations(entry, desc or "")
            if flags:
                st.warning("Potential hallucinated mentions: " + ", ".join(flags))
            else:
                st.success("No obvious hallucinated object mentions detected (heuristic).")
        if show_debug_prompt and debug_prompt:
            try:
                dp = debug_prompt(entry)
                with st.expander("Debug prompt passed to LM"):
                    st.code(dp)
            except Exception as e:
                st.write("debug_prompt error:", e)
        if st.button("Save result to data/test_outputs/last_result.json"):
            os.makedirs("data/test_outputs", exist_ok=True)
            out = {"entry": entry, "generated": desc}
            with open("data/test_outputs/last_result.json", "w") as fh:
                json.dump(out, fh, indent=2)
            st.success("Saved to data/test_outputs/last_result.json")

if run_refine:
    if refine_fn is None:
        st.error("refine_description not available in description_generator.")
    else:
        try:
            base = gen_fn(entry) if gen_fn else ""
        except Exception as e:
            st.error("Could not generate base description for refinement: " + str(e)); base = ""
        with st.spinner("Refining..."):
            try:
                revised = refine_fn(entry, base, user_instruction)
            except Exception as e:
                st.error("Refinement error: " + str(e)); revised = ""
        st.markdown("### Refined Description")
        st.write(revised or "(empty)")
        if show_halluc:
            flags = detect_hallucinations(entry, revised or "")
            if flags:
                st.warning("Potential hallucinated mentions after refinement: " + ", ".join(flags))
            else:
                st.success("No obvious hallucinated mentions after refinement (heuristic).")
        if st.button("Save refined result to data/test_outputs/refined_result.json"):
            os.makedirs("data/test_outputs", exist_ok=True)
            out = {"entry": entry, "generated": revised, "instruction": user_instruction}
            with open("data/test_outputs/refined_result.json", "w") as fh:
                json.dump(out, fh, indent=2)
            st.success("Saved to data/test_outputs/refined_result.json")

st.markdown("---")
st.markdown("Developer notes: This testbench surfaces the cleaned/merged facts from description_generator._serialize_facts (if present). Use the JSON editor to tweak the graph and re-run generation to see how the cleaned facts and description change.")
