# Urban Scene Intelligence

Static urban scene understanding pipeline: upload an image, detect objects, build a semantic scene graph, generate an initial narrative description, and optionally refine it (interactive or seq2seq).

---

## Quick start

1. Create & activate virtualenv (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit demo
```powershell
streamlit run app.py
```

3. Git workflow (one-time repo init)
```powershell
git init
git add .
git commit -m "chore: initial commit"
git checkout -b feature/add-readme-docs
```

---

## High-level pipeline (stages)

1. Stage 1 — Object detection
   - Input: image file (app saves uploaded image to `data/uploaded_image.jpg`).
   - Model: YOLOv8 (small weights commonly: `yolov8s.pt`).
   - Output: list of detection dictionaries. Each detection typically contains:
     - `label` (string), `bbox` (x,y,w,h), `confidence` (float), optional `attributes`.
   - Primary module: `stage1_object_detection.py` → `run_object_detection(image_path, save_output=False)`.
   - Libraries: ultralytics / YOLOv8, torch, opencv, PIL.

2. Stage 2 — Semantic Scene Graph construction
   - Input: detections (from Stage 1) + optional scene attributes (weather, time_of_day).
   - Output: scene graph (project uses two interchange forms; canonical schema recommended below):
     - dict with keys:
       - `nodes`: list of { `id`, `label`, `attributes`: [..], optional `bbox` }
       - `edges`: list of { `source`, `target`, `relation` }
       - `scene_attributes`: { `weather`, `time_of_day`, ... }
     - Some scripts may return a `networkx.Graph`/`DiGraph`.
   - Primary module: `stage2_semantic_scene_graph.py` → `build_scene_graph(...)`, `save_scene_graph(...)`.
   - Libraries: networkx, shapely (optional), numpy.

3. Stage 3 — Description generation
   - Input: canonical scene graph dict.
   - Output: human-readable narrative description (string).
   - Primary module: `description_generator.py` → `generate_narrative_description(graph)`.
   - Implementation: rules / template-based and small LLM prompts. Can be replaced by generative models for better diversity.

4. Stage 4 — Refinement (optional)
   - Two modes:
     - Streamlit one-shot refinement: UI appends user focus and re-runs `generate_narrative_description`.
     - Seq2seq refinement using Flan-T5: build textual prompt from `scene_graph_to_text` and call `model.generate(...)`.
   - Primary modules: `stage4_refinement.py`, `stage4_interactive_refinement.py`.
   - Libraries: transformers, torch.

---

## Project structure (current repository)

- `app.py`
  - Streamlit UI: upload, stage orchestration, one-round refinement UI.
- `stage1_object_detection.py`
  - YOLOv8 wrapper (detection + visualization).
- `stage2_semantic_scene_graph.py`
  - Build, save, and convert scene graph representations; graph → text helpers.
- `description_generator.py`
  - Map scene graph → narrative text.
- `stage4_refinement.py` / `stage4_interactive_refinement.py`
  - Seq2seq refinement (Flan-T5), interactive CLI.
- `data/`
  - Runtime artifacts: uploaded images, saved outputs.
- `yolov8s.pt` (if present)
  - YOLO weights — avoid committing large files; prefer download script or git-lfs.
- `requirements.txt`
  - Python package requirements.

---

## Canonical scene graph schema (recommended)

Use a simple JSON-serializable format so all modules and LLM prompts can consume the same representation.

Example:
```json
{
  "nodes": [
    {"id": 1, "label": "car", "attributes": ["red", "parked"], "bbox": [x,y,w,h]},
    {"id": 2, "label": "pedestrian", "attributes": ["crossing"], "bbox": [x,y,w,h]}
  ],
  "edges": [
    {"source": 2, "target": 1, "relation": "in front of"}
  ],
  "scene_attributes": {"weather": "sunny", "time_of_day": "daytime"}
}
```

Add helper converters:
- `nx_to_dict(nx_graph)` and `dict_to_nx(graph_dict)` to normalize in-code exchange.

---

## Key libraries & models

- Python packages (examples; see `requirements.txt` for pinned versions):
  - streamlit, pillow, matplotlib
  - ultralytics (YOLOv8) or YOLOv8 wrapper
  - torch
  - transformers (for Flan-T5 or other seq2seq)
  - networkx
  - numpy, opencv-python, shapely (optional)
- Models:
  - YOLOv8 weights (e.g., `yolov8s.pt`) — detection.
  - Flan-T5 (or other T5 variant) — refinement (optional).
  - Any external LLM used via API would be integrated in refinement stage.

---

## How the data flows

Upload image (Streamlit) → saved to `data/uploaded_image.jpg` → Stage1 detects objects → Stage2 builds scene graph → Stage3 generates narrative → Stage4 optionally refines description with user instruction or seq2seq model.

---

## Testing & UI usage

This project includes a Streamlit demo (`app.py`) and a small test script `test_graph_app.py` that exercises the scene-graph -> description flow. Below are instructions to run, test, and interact with the UI.

1) Prepare your environment

```powershell
# create & activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Run the Streamlit app (manual UI testing)

```powershell
streamlit run app.py
```

How to interact with the UI
- Open the URL printed by Streamlit (usually http://localhost:8501).
- Use the sidebar file uploader to pick an image (`.jpg`, `.jpeg`, `.png`).
- The app saves the uploaded image to `data/` and runs the detection + scene graph + description pipeline automatically. Watch the app output for each stage.
- If you want to test with prepared data instead of uploading, copy an image into `data/test_images/` and modify `app.py` to load that file path directly (search for the `uploaded_file` handling section and replace with a hardcoded path), or use the 'Choose from sample images' UI if present.

3) Run the test script (automated test)

```powershell
# from project root
python test_graph_app.py
```

What the test does
- `test_graph_app.py` will load a sample graph (`data/sample_entry1.json`) or a sample image from `data/test_images/` depending on its implementation. It runs the scene graph -> description generator and prints the output.
- Use this to validate that changes to `description_generator.py` or `stage2_semantic_scene_graph.py` preserve expected behavior.

4) Troubleshooting and tips
- If Streamlit fails to start, make sure no other process is using port 8501 or run `streamlit run app.py --server.port 8502` to select a different port.
- If object detection fails (missing weights), download YOLO weights to the project root (or update `stage1_object_detection.py` to point to an external path). Consider adding `scripts/download_weights.ps1` to automate this.
- For CI, run `python -m pytest -q` after adding unit tests.

## Running, debugging & common pitfalls

- Not a git repo: run `git init` then commit before creating branches.
- Remote repo errors: create the GitHub repo first; then `git remote add origin <url>` and push.
- Graph schema mismatch: some modules expect `nx.Graph`, others expect dict — add conversion helpers to avoid runtime errors.
- Large model files: do not commit `.pt` files. Add them to `.gitignore` and provide `scripts/download_weights.ps1` to fetch them.
- GPU: YOLO and T5 inference are much faster with CUDA-enabled GPU; CPU inference is possible but slow.

---

## Recommended improvements (short list)

- Normalize scene graph format and add conversion helpers (`usi/utils/graph_utils.py`).
- Move scripts into a package directory (`usi/` or `src/usi`) for cleaner imports and testing.
- Add unit tests for graph conversion, description generator, and a mocked detection wrapper.
- Add `.gitignore`, `.gitattributes` (if using git-lfs), and a GitHub Actions CI workflow.
- Replace committed weights with download scripts or git-lfs.
- Add example inputs/expected outputs in `examples/` for LLM prompt testing.

---

## Contributing

- Branch naming: `feature/*`, `bugfix/*`, `docs/*`.
- Workflow:
  - Create feature branch, implement, commit, push, open PR into `main`.
  - Include unit tests for new logic.
- Keep venv (e.g., `.venv/`) in `.gitignore`.

---

## License & attribution

Add a LICENSE file appropriate to your needs (MIT recommended for permissive use).

---

If you want, I will:
- Create this README file on a new branch and show the exact git commands to commit & push.
- Add `.gitignore` and a small `scripts/download_weights.ps1`.
