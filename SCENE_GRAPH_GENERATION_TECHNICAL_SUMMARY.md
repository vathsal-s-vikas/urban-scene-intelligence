# Urban Scene Intelligence: Scene Graph Generation Technical Summary

## Project Overview
**Urban Scene Intelligence** is an application that analyzes urban scenes to generate structured scene graphs representing objects, their relationships, and spatial arrangements. The system combines deep learning models (RELTR) with visual transformers (CLIP) to create comprehensive semantic representations of urban environments.

---

## Architecture & Component Overview

### Core Components
1. **Backbone: ResNet-50 with Position Encoding**
2. **Transformer: Dual-path Attention Mechanism (RelTR)**
3. **Visual Features: CLIP for Attribute Extraction**
4. **Scene Graph Format: Visual Genome-Compatible JSON**

---

## Detailed Pipeline: Image to Scene Graph

### Phase 1: Input Processing & Model Initialization

#### 1.1 Image Input Handling
- **Input Flexibility**: Accepts both PIL Images and PyTorch Tensors
- **PIL Image Processing**:
  - Resize to 800px (maintaining aspect ratio via `T.Resize(800)`)
  - Convert to tensor: `T.ToTensor()`
  - Normalize using ImageNet statistics: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
  - Add batch dimension: unsqueeze(0) → shape `[1, 3, H, W]`
  - **Original image dimensions preserved**: stored in `self.original_size = (width, height)` from PIL `.size` property

- **Tensor Input Processing**:
  - 3D Tensor (C, H, W): Extract size from shape[2], shape[1]
  - 4D Tensor (B, C, H, W): Already batched, use as-is
  - Extract original size: `(shape[3], shape[2])` for (width, height)

#### 1.2 Model Architecture Construction
**Model**: `RelTR` (Relation Transformer for Scene Graph Generation)

**Architecture Components**:

a) **Backbone**: ResNet-50 with Position Encoding
   - Configuration: `Backbone('resnet50', False, False, False)`
   - Position Encoding: `PositionEmbeddingSine(128, normalize=True)`
   - Combined via `Joiner` class
   - Output channels: 2048

b) **Transformer**:
   - **Encoder Layers**: 6
   - **Decoder Layers**: 6
   - **Model Dimension (d_model)**: 256
   - **Attention Heads (nhead)**: 8
   - **Feed-Forward Dimension**: 2048
   - **Dropout**: 0.1
   - **Return Intermediate**: True (for auxiliary losses)

c) **RelTR Model Configuration**:
   - **Number of Classes (entities)**: 151 (Visual Genome entity set)
   - **Number of Relationship Classes**: 51
   - **Number of Entity Queries**: 100
   - **Number of Triplet Queries**: 200 (for subject-predicate-object triplets)

#### 1.3 Model Checkpoint Loading
- **Checkpoint Path**: `./reltr/checkpoint0149.pth`
- **Loading Method**: `torch.load(output_path, map_location='cpu', weights_only=False)`
- **Model State**: Loaded via `self.model.load_state_dict(ckpt['model'])`
- **Inference Mode**: `self.model.eval()`

---

### Phase 2: Forward Pass & Feature Extraction

#### 2.1 Hook Installation (Intermediate Feature Capture)
**Purpose**: Capture intermediate representations for visualization and analysis

Three forward hooks registered:
1. **Backbone Feature Hook** (Convolutional Features):
   - Target: `self.model.backbone[-2]` (final convolutional layer)
   - Captures: Feature maps (output shape varies with input, typically `[1, 2048, H', W']`)
   - Usage: Baseline for attention map overlays

2. **Cross-Attention Subject Hook**:
   - Target: `self.model.transformer.decoder.layers[-1].cross_attn_sub` (final decoder layer, subject attention)
   - Captures: Attention weights `output[1]` from cross-attention mechanism
   - Shape: `[1, num_queries, H'*W']` where H', W' are feature map dimensions
   - Usage: Visualizes which image regions the model attends to when predicting subjects

3. **Cross-Attention Object Hook**:
   - Target: `self.model.transformer.decoder.layers[-1].cross_attn_obj` (final decoder layer, object attention)
   - Captures: Attention weights `output[1]` from cross-attention mechanism
   - Shape: `[1, num_queries, H'*W']`
   - Usage: Visualizes which image regions the model attends to when predicting objects

**Hook Lifecycle**: Registered before forward pass, removed after capturing.

#### 2.2 Forward Pass
```
Model Input: self.img [1, 3, 800, 800] (normalized)
     ↓
[Backbone + Position Encoding]
     ↓
[Transformer Encoder/Decoder]
     ↓
Model Outputs Dictionary
```

**Output Tensors** (`self.outputs`):
- `sub_boxes`: Subject bounding boxes, shape `[1, num_queries, 4]`, format `[cx, cy, w, h]` (normalized 0-1)
- `sub_logits`: Subject class logits, shape `[1, num_queries, 152]` (151 classes + background)
- `obj_boxes`: Object bounding boxes, shape `[1, num_queries, 4]`, format `[cx, cy, w, h]` (normalized 0-1)
- `obj_logits`: Object class logits, shape `[1, num_queries, 152]`
- `rel_logits`: Relationship class logits, shape `[1, num_queries, 52]` (51 relationships + background)

**Captured Intermediate Features**:
- `self.conv_features`: Backbone feature maps
- `self.dec_attn_weights_sub`: Subject cross-attention weights
- `self.dec_attn_weights_obj`: Object cross-attention weights

---

### Phase 3: Confidence Filtering & Prediction Selection

#### 3.1 Softmax and Confidence Extraction
```python
probas = self.outputs['rel_logits'].softmax(-1)[0, :, :-1]          # [num_queries, 51]
probas_sub = self.outputs['sub_logits'].softmax(-1)[0, :, :-1]      # [num_queries, 151]
probas_obj = self.outputs['obj_logits'].softmax(-1)[0, :, :-1]      # [num_queries, 151]
```
- Each `probas_*` contains normalized probability scores for each class
- `:-1` excludes background class for cleaner predictions

#### 3.2 Confidence Threshold Filtering
- **Threshold**: 0.3 (30% minimum confidence)
- **Filtering Logic**: Keep predictions where ALL three components (relation, subject, object) exceed threshold
```
self.keep = (probas.max(-1).values > 0.3) AND 
            (probas_sub.max(-1).values > 0.3) AND 
            (probas_obj.max(-1).values > 0.3)
```
- **Result**: `self.keep` is a boolean mask of shape `[num_queries]`

#### 3.3 Prediction Ranking & Top-K Selection
```python
self.keep_queries = torch.nonzero(self.keep, as_tuple=True)[0]  # Indices of kept predictions
```
- Extract indices where `self.keep == True`

**Ranking by Confidence Product**:
```python
self.indices = torch.argsort(-probas[self.keep_queries].max(-1)[0] * 
                             probas_sub[self.keep_queries].max(-1)[0] * 
                             probas_obj[self.keep_queries].max(-1)[0])[:topk]
```
- Multiply max probabilities across all three components
- Sort in descending order (negative sign)
- Keep top 30 predictions (`topk = 30`)
- **Result**: `self.keep_queries` contains indices of top-30 highest-confidence triplets

#### 3.4 Probability Storage
```python
self.probas = probas
self.probas_sub = probas_sub
self.probas_obj = probas_obj
```
- Stored for later use in object class prediction and visualization

---

### Phase 4: Bounding Box Coordinate Transformation

#### 4.1 Coordinate Format Conversion
RelTR outputs boxes in **normalized center format** `[cx, cy, w, h]` where all values are in range [0, 1].

**Conversion Function**: `box_cxcywh_to_xyxy(x)`
```
Input:  [cx, cy, w, h] (normalized)
Output: [x1, y1, x2, y2] (normalized corner format)

x1 = cx - 0.5 * w
y1 = cy - 0.5 * h
x2 = cx + 0.5 * w
y2 = cy + 0.5 * h
```
- Still in normalized space [0, 1]

#### 4.2 Rescaling to Image Coordinates
**Function**: `rescale_bboxes(out_bbox, size)`
```
Input:  Bounding boxes [x1, y1, x2, y2] (normalized), size = (width, height)
Output: Bounding boxes [x1, y1, x2, y2] (pixel coordinates)

Scaling: multiply by [img_w, img_h, img_w, img_h]
```

**Key Coordinate System Handling**:
- Model operates on resized image (800px)
- Original image dimensions preserved: `self.original_size = (PIL_width, PIL_height)`
- **Rescaling in build_scene_graph()**:
  ```python
  sub_bboxes_scaled = self.rescale_bboxes(
      self.outputs['sub_boxes'][0, self.keep],  # All kept predictions
      self.original_size                         # (width, height) from original PIL Image
  )
  obj_bboxes_scaled = self.rescale_bboxes(
      self.outputs['obj_boxes'][0, self.keep],
      self.original_size
  )
  ```
- Result: `[N, 4]` tensors in pixel coordinates of original image

#### 4.3 Final Bounding Box Format for Scene Graph
**Conversion**: `bbox_to_xywh(bbox)` where bbox = `[x1, y1, x2, y2]`
```python
def bbox_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)
```
- Converts corner format to top-left + size format: `[x, y, width, height]`
- All values are floats in pixel coordinates of original image
- **Stored in scene graph objects as**: `x`, `y`, `w`, `h` fields

---

### Phase 5: Object Deduplication & Merging

#### 5.1 IoU-Based Matching (Before Storing in Scene Graph)
**Purpose**: Prevent duplicate objects from being stored as separate entities when detected multiple times with high overlap.

**Method**: `find_matching_object(obj_class, bbox_normalized, normalized_objects)`

- **Input**:
  - `obj_class`: Predicted object class name (e.g., "car", "person")
  - `bbox_normalized`: Bounding box in normalized space `[cx, cy, w, h]`
  - `normalized_objects`: Already-processed objects with their normalized bounding boxes

- **Matching Strategy (Class-Dependent)**:

  a) **Large Architectural Elements** (building, street, wall, floor, ground, ceiling):
     - Use **containment heuristic**: `is_contained_or_contains(bbox1, bbox2, threshold=0.4)`
     - One box being largely contained in another indicates same entity
     - Threshold: 40% of smaller area must overlap with larger area

  b) **Architectural Details** (window, door, sign, light):
     - Use **area similarity**: Check if bbox areas differ by factor < 2.5
     - If similar area and same class → likely same object
     - Threshold: `area_ratio_threshold = 2.5`

  c) **Regular Objects** (car, person, tire, bicycle, etc.):
     - Use **standard IoU (Intersection over Union)**
     - Formula:
       ```
       IoU = intersection_area / union_area
       ```
     - Default threshold: 0.5 (50% overlap)

#### 5.2 IoU Calculation
**Function**: `calculate_iou(bbox1, bbox2)` where bboxes are in `[x1, y1, x2, y2]` format
```
intersection_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * 
                    max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
union_area = bbox1_area + bbox2_area - intersection_area

IoU = intersection_area / union_area if union_area > 0 else 0.0
```

#### 5.3 Containment Check
**Function**: `is_contained_or_contains(bbox1, bbox2, threshold=0.4)`
- Computes intersection area and normalized containment ratios for both boxes
- Returns True if either box contains the other with >= 40% overlap
- Used for large architectural elements where one object might genuinely contain another

#### 5.4 Object Merging (Post-Processing)
**Function**: `merge_similar_objects(scene_graph, iou_thresh=0.4)`

Applied after initial scene graph construction:
- Identifies duplicate objects across all stored objects
- For each pair with IoU > 0.4 and matching class names:
  - Keep one as primary
  - Merge attributes from duplicates
  - Map deleted object IDs to primary object ID
- Update all relationships to point to primary object IDs
- Remove duplicate relationships (same subject-predicate-object triplet)
- Deduplicate relationship list

---

### Phase 6: Scene Graph Construction

#### 6.1 Object Dictionary Format (Visual Genome Compliant)
```json
{
  "object_id": 1,
  "names": ["car"],           // Single element list (primary name)
  "synsets": [],              // Empty (can be extended)
  "x": 150.5,                 // Top-left x coordinate (pixels)
  "y": 200.3,                 // Top-left y coordinate (pixels)
  "w": 300.2,                 // Width (pixels)
  "h": 150.8,                 // Height (pixels)
  "attributes": []            // Empty initially (filled by CLIP later)
}
```

#### 6.2 Relationship Dictionary Format (Visual Genome Compliant)
```json
{
  "relationship_id": 1,
  "subject_id": 1,            // ID of subject object
  "object_id": 2,             // ID of object entity
  "predicate": "parked on",   // Relationship type from REL_CLASSES
  "synsets": []               // Empty (can be extended)
}
```

#### 6.3 Relationship Class Set (51 Classes)
```
['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 
'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 
'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 
'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 
'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 
'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 
'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
```

#### 6.4 Entity Class Set (151 Classes)
Full set includes common urban scene objects: airplane, animal, arm, bag, banana, basket, beach, bear, bed, bench, bike, bird, board, boat, book, boot, bottle, bowl, box, boy, branch, building, bus, cabinet, cap, car, cat, chair, child, clock, coat, counter, cow, cup, curtain, desk, dog, door, drawer, ear, elephant, engine, eye, face, fence, finger, flag, flower, food, fork, fruit, giraffe, girl, glass, glove, guy, hair, hand, handle, hat, head, helmet, hill, horse, house, jacket, jean, kid, kite, lady, lamp, laptop, leaf, leg, letter, light, logo, man, men, motorcycle, mountain, mouth, neck, nose, number, orange, pant, paper, paw, people, person, phone, pillow, pizza, plane, plant, plate, player, pole, post, pot, racket, railing, rock, roof, room, screen, seat, sheep, shelf, shirt, shoe, short, sidewalk, sign, sink, skateboard, ski, skier, sneaker, snow, sock, stand, street, surfboard, table, tail, tie, tile, tire, toilet, towel, tower, track, train, tree, truck, trunk, umbrella, vase, vegetable, vehicle, wave, wheel, window, windshield, wing, wire, woman, zebra

#### 6.5 Scene Graph Construction Loop
For each top-K prediction (up to 30):
1. **Extract Predicted Classes**:
   - Relationship class: `REL_CLASSES[probas[idx].argmax()]`
   - Subject class: `CLASSES[probas_sub[idx].argmax()]`
   - Object class: `CLASSES[probas_obj[idx].argmax()]`

2. **Check for Existing Objects** (using IoU matching):
   - Query: Does an object of this class with similar bbox already exist?
   - If YES: Reuse existing `object_id`
   - If NO: Create new object with fresh `object_id`, increment counter

3. **Add Relationship**:
   - Only if triplet (subject_id, predicate, object_id) doesn't already exist
   - Prevents duplicate relationships from multiple detections

4. **Update Counters**:
   - `next_object_id`: Incremented when new object created
   - `next_relationship_id`: Incremented when new relationship added

#### 6.6 Final Scene Graph JSON Structure
```json
{
  "image_id": 1,
  "objects": [
    {"object_id": 1, "names": ["car"], "synsets": [], "x": 150.5, "y": 200.3, "w": 300.2, "h": 150.8, "attributes": []},
    {"object_id": 2, "names": ["street"], "synsets": [], "x": 0.0, "y": 0.0, "w": 800.0, "h": 600.0, "attributes": []},
    ...
  ],
  "relationships": [
    {"relationship_id": 1, "subject_id": 1, "object_id": 2, "predicate": "parked on", "synsets": []},
    ...
  ]
}
```

#### 6.7 File Saving
- **Output Path**: `data/{image_name}_scene_graph.json`
- **Format**: JSON with 4-space indentation for readability
- **Cache Logic**: If file already exists, skip processing and load cached version

---

### Phase 7: Visual Attribute Extraction (CLIP Integration)

#### 7.1 CLIP Model Setup
- **Model Source**: HuggingFace Transformers
- **Model Path**: Local CLIP checkpoint directory
- **Processor**: CLIPProcessor for image-text encoding
- **Device**: CUDA if available, else CPU

#### 7.2 Pre-computed Attribute Embeddings
- **Embeddings File**: `clip_attribute_embeddings.pt`
- **Contents**:
  - `embeddings`: Text embeddings for attribute vocabulary (shape `[vocab_size, embedding_dim]`)
  - `vocab`: List of attribute strings (e.g., ["red", "parked", "large", ...])

#### 7.3 Attribute Extraction Process
For each object in scene graph:
1. **Extract Object Crop**:
   - Denormalize image tensor using ImageNet statistics
   - Crop region from image: `[x, y, x+w, y+h]`

2. **CLIP Image Encoding**:
   - Process crop through CLIP image encoder
   - Get image feature vector (embedding)
   - L2-normalize: `image_embed /= image_embed.norm(dim=-1, keepdim=True)`

3. **Similarity Matching**:
   - Compute cosine similarity: `sims = image_embed @ text_embeds.T`
   - Result: vector of similarities to all attribute vocabulary words

4. **Threshold Filtering**:
   - **Similarity Threshold**: 0.18 (minimum acceptable match strength)
   - Select top-5 attributes (`TOP_K = 5`)
   - Keep only attributes with `similarity > 0.18`

5. **Store Attributes**:
   - Update object in scene graph: `obj["attributes"] = top_attrs`

#### 7.4 Output
- **Enhanced Scene Graph File**: `data/{image_name}_scene_graph_with_attributes.json`
- Same structure as base scene graph but with populated `attributes` field

---

## Data Flow Diagram

```
Input Image (PIL)
       ↓
[Resize 800px + Normalize]
       ↓
[Store original size]
       ↓
       ├──→ [Feed to RelTR Model]
       │         ↓
       │    [Backbone (ResNet-50) + Positional Encoding]
       │         ↓
       │    [Transformer Encoder/Decoder]
       │         ↓
       │    [Hook 1: Conv Features]
       │    [Hook 2: Subject Attention]
       │    [Hook 3: Object Attention]
       │         ↓
       │    Output: sub_boxes, sub_logits, obj_boxes, obj_logits, rel_logits
       │
       ├──→ [Confidence Filtering (0.3 threshold)]
       │         ↓
       │    Keep predictions with high confidence across all components
       │         ↓
       │    [Ranking & Top-K Selection (top 30)]
       │         ↓
       │    Sorted by confidence product
       │
       ├──→ [Coordinate Transformation]
       │         ├──→ box_cxcywh_to_xyxy (normalized)
       │         └──→ rescale_bboxes (to original image pixels)
       │
       ├──→ [Object Deduplication]
       │         ├──→ IoU Matching (for regular objects)
       │         ├──→ Containment Check (for large structures)
       │         └──→ Area Similarity (for architectural details)
       │
       ├──→ [Scene Graph Construction]
       │         ├──→ Create/Reuse Objects
       │         ├──→ Add Relationships
       │         └──→ Avoid Duplicates
       │
       ├──→ [Post-Processing Merge]
       │         ├──→ Merge Similar Objects
       │         ├──→ Update Relationship IDs
       │         └──→ Deduplicate Relationships
       │
       ├──→ [Save Base Scene Graph]
       │         └──→ data/{image_name}_scene_graph.json
       │
       └──→ [CLIP Attribute Extraction]
              ├──→ Denormalize & Crop Objects
              ├──→ Encode via CLIP
              ├──→ Match to Attribute Vocabulary
              └──→ Save Enhanced Scene Graph
                     └──→ data/{image_name}_scene_graph_with_attributes.json
```

---

## Key Technical Specifications

| Component | Specification |
|-----------|---------------|
| **Backbone** | ResNet-50 |
| **Position Encoding** | PositionEmbeddingSine (dim=128) |
| **Transformer Encoder Layers** | 6 |
| **Transformer Decoder Layers** | 6 |
| **Model Dimension (d_model)** | 256 |
| **Attention Heads** | 8 |
| **Feed-Forward Dimension** | 2048 |
| **Entity Classes** | 151 |
| **Relationship Classes** | 51 |
| **Entity Queries** | 100 |
| **Triplet Queries** | 200 |
| **Confidence Threshold** | 0.3 (30%) |
| **Top-K Predictions** | 30 |
| **Input Image Size** | Resized to 800px |
| **Bounding Box Format (Model)** | [cx, cy, w, h] (normalized 0-1) |
| **Bounding Box Format (Scene Graph)** | [x, y, w, h] (pixels) |
| **IoU Threshold (Regular Objects)** | 0.5 |
| **Containment Threshold (Large Structures)** | 0.4 |
| **Area Ratio Threshold (Details)** | 2.5 |
| **CLIP Attribute Top-K** | 5 |
| **CLIP Similarity Threshold** | 0.18 |

---

## Important Implementation Details

### Coordinate System Handling
- **Model Processing**: Image resized to 800px, all predictions in normalized [0, 1]
- **Original Dimensions Preservation**: PIL `.size` gives (width, height) in original resolution
- **Rescaling Formula**: `pixel_coords = normalized_coords * [img_width, img_height, img_width, img_height]`
- **Scene Graph Storage**: All coordinates in pixel space of original image dimensions

### Deduplication Logic Priority
1. Check class name match (same entity type)
2. Compute IoU or use class-specific heuristic
3. If threshold exceeded → merge, reuse object_id
4. Track mapping of deleted IDs to primary IDs
5. Update all relationships to use primary IDs
6. Remove duplicate relationships

### Confidence Filtering
- **Multi-component AND condition**: All three (relation, subject, object) must exceed 0.3
- **Prevents low-confidence relationships** from cluttering scene graph
- **Ranking Metric**: Product of max probabilities: `P(rel) × P(sub) × P(obj)`
  - Favors predictions where model is confident about all aspects
  - Geometric mean would also be reasonable: `(P(rel) × P(sub) × P(obj))^(1/3)`

### Hook Mechanism
- **Purpose**: Capture intermediate activations without modifying forward pass
- **Timing**: Hooks fire during forward pass, captured data accessible after
- **Cleanup**: Manually removed post-forward to prevent memory leaks
- **Usage**: Enables attention visualization without computational overhead

### Caching Strategy
- **Input File Check**: Before processing, check if `data/{image_name}_scene_graph.json` exists
- **Avoid Reprocessing**: If found, load and return cached version
- **Efficiency**: Significant speedup for repeated image analysis

---

## Output Schema

### Base Scene Graph (`_scene_graph.json`)
```json
{
  "image_id": 1,
  "objects": [
    {
      "object_id": <int>,
      "names": [<string>],
      "synsets": [],
      "x": <float>,
      "y": <float>,
      "w": <float>,
      "h": <float>,
      "attributes": []
    }
  ],
  "relationships": [
    {
      "relationship_id": <int>,
      "subject_id": <int>,
      "object_id": <int>,
      "predicate": <string>,
      "synsets": []
    }
  ]
}
```

### Enhanced Scene Graph (`_scene_graph_with_attributes.json`)
Same schema as above, but `attributes` field populated with visual attribute strings extracted by CLIP.

---

## Performance Considerations

- **Model Size**: RelTR checkpoint ≈ 150MB (checkpoint0149.pth)
- **Inference Time**: ~5-10 seconds per image (GPU dependent, single-image processing)
- **Memory Usage**: 
  - Model: ~400MB (VRAM)
  - Intermediate features (hooks): Minimal additional overhead
  - Feature maps for visualization: ≈ 50-100MB
- **GPU Requirement**: CUDA-capable device recommended (falls back to CPU if unavailable)
- **Caching Benefit**: Second analysis of same image loads in < 1 second

---

## Summary of Key Innovations in This Implementation

1. **Class-Aware Deduplication**: Different matching strategies for different object types (architectural vs. regular vs. details)
2. **Normalized Coordinate Space Preservation**: Maintains normalized bboxes during matching, rescales only for final output
3. **Multi-component Confidence Filtering**: Ensures all three relationship components (predicate, subject, object) are confident
4. **Hook-based Feature Capture**: Enables efficient visualization without modifying model architecture
5. **Visual Genome Compliance**: Output format fully compatible with Visual Genome dataset standards
6. **Two-stage Processing**: Separate scene graph generation (structural) and attribute extraction (visual) phases

---

## Next Phase (Not Covered Here)

After scene graph generation, the system proceeds to natural language description generation using the complete scene graph with attributes. This involves:
- Text enrichment and connectors
- Object categorization and salience ranking
- Narrative-style description generation
- See: `description_generator.py`

