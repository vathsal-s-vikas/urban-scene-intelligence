# Urban Scene Intelligence 🏙️

A sophisticated urban scene understanding system that generates detailed descriptions of urban environments using advanced computer vision and natural language processing. The system employs RELTR (Relationship Transformer) for object detection and relationship understanding, enhanced by CLIP (Contrastive Language-Image Pre-Training) for rich visual attribute extraction, creating a comprehensive scene understanding pipeline.

## Features

- 🔍 **Object Detection & Relationship Understanding**: Uses RELTR to identify objects and their relationships in urban scenes
- 🎨 **Visual Attribute Extraction**: Leverages CLIP to enrich objects with detailed visual attributes
- �️ **Scene Graph Generation**: Creates comprehensive scene graphs with objects, relationships, and attributes
- �️ **Attention Visualization**: Provides insights into how RELTR understands object relationships through attention maps
- �📝 **Natural Language Description**: Generates human-readable descriptions from scene graphs
- 🔄 **Description Refinement**: Supports interactive refinement of generated descriptions

## Prerequisites

- Python 3.6 or higher
- CUDA-capable GPU (recommended for faster processing)
- Windows/Linux operating system

## Quick start

1. Create & activate virtualenv (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Download required models
```powershell
# Download CLIP model and embeddings
.\scripts\download_models.ps1
```

3. Run the Streamlit demo
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

## Project Components

### 1. Core Scene Understanding
- **RELTR (Relationship Transformer)**: Primary model for object detection and relationship understanding
- **CLIP (Contrastive Language-Image Pre-Training)**: Visual attribute extraction and enhancement
- **ResNet50**: Backbone network for feature extraction in RELTR

### 2. Scene Graph Generation (`scenegraph.py`)
- Integrates RELTR and CLIP models
- Creates comprehensive scene graphs with:
  - Objects and their locations
  - Spatial relationships between objects
  - Visual attributes for each object
  - Confidence scores for detections

### 3. Description Generation (`description_generator.py`)
- Converts scene graphs to natural language
- Implements template-based generation
- Handles complex spatial relationships
- Produces coherent scene descriptions

### 4. Interactive Refinement (`stage4_refinement.py`)
- Allows description refinement using T5 models
- Supports focus on specific scene aspects
- Interactive or automated refinement options

---

## Project Structure

```
urban-scene-intelligence/
├── app.py                      # Streamlit web interface
├── scenegraph.py              # Core scene graph generation
├── description_generator.py    # Natural language generation
├── stage4_refinement.py       # Description refinement
├── data/                      # Runtime data storage
│   ├── test_images/          # Sample images
│   └── scene_graph.json      # Generated graphs
├── scripts/                   # Utility scripts
│   ├── download_clip.py      # CLIP model download
│   └── download_models.ps1   # Model download orchestration
├── clip_model/               # CLIP model files
├── reltr/                    # RELTR model and weights
│   └── checkpoint0149.pth    # RELTR model checkpoint
└── requirements.txt          # Python dependencies

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
  - RelTR for scene graph generation
  - Flan-T5 (or other T5 variant) — refinement (optional).
  - Any external LLM used via API would be integrated in refinement stage.

---

### Environment Setup

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