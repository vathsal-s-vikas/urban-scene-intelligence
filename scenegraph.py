# stage2_scene_graph_generator.py
import json
from xml.parsers.expat import model
import numpy as np
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel

from PIL import Image
import requests
import matplotlib.pyplot as plt
from reltr.models.backbone import Backbone, Joiner
from reltr.models.position_encoding import PositionEmbeddingSine
from reltr.models.transformer import Transformer
from reltr.models.reltr import RelTR
import os
import torch # Import torch here
from tqdm import tqdm
import gc

class SceneGraphGenerator:
    def __init__(self):
        self.model = None
        self.outputs = None
        self.img = None
        self.keep = None
        self.indices = None
        self.keep_queries = None
        self.original_size = None  # To store original image dimensions
        # Store intermediate features
        self.conv_features = None
        self.dec_attn_weights_sub = None
        self.dec_attn_weights_obj = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
        self.REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
              (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @staticmethod
    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = SceneGraphGenerator.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        Bounding boxes are in format [x1, y1, x2, y2]
        """
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection area
        intersect_x_min = max(x1_min, x2_min)
        intersect_y_min = max(y1_min, y2_min)
        intersect_x_max = min(x1_max, x2_max)
        intersect_y_max = min(y1_max, y2_max)

        # Check if there's an intersection
        if intersect_x_max < intersect_x_min or intersect_y_max < intersect_y_min:
            return 0.0

        intersect_area = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)

        # Calculate union area
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - intersect_area

        # Calculate IoU
        iou = intersect_area / union_area if union_area > 0 else 0.0

        return iou

    # Default value for containment is 0.4
    # Initially was set to 0.6
    def is_contained_or_contains(self, bbox1, bbox2, threshold=0.4):
        """Check if one bbox is largely contained within another (for normalized coords)."""
        x1_1, y1_1, x1_max, y1_max = bbox1
        x1_2, y1_2, x2_max, y2_max = bbox2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)

        if xi2 < xi1 or yi2 < yi1:
            return False

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x1_max - x1_1) * (y1_max - y1_1)
        area2 = (x2_max - x1_2) * (y2_max - y1_2)

        containment1 = inter_area / area1 if area1 > 0 else 0
        containment2 = inter_area / area2 if area2 > 0 else 0

        return max(containment1, containment2) >= threshold

    def find_matching_object(self, obj_class, bbox_normalized, normalized_objects):
        """
        Find matching object using NORMALIZED coordinates (before rescaling).
        Uses class-specific heuristics for better matching.

        Args:
            obj_class: Class name of the object
            bbox_normalized: Bounding box in normalized [0,1] space [cx, cy, w, h]
            normalized_objects: List of objects with normalized bboxes
        """
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        cx, cy, w, h = bbox_normalized
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        bbox_xyxy = [x1, y1, x2, y2]
        bbox_area = w * h

        # Strategy depends on object class
        if obj_class in ['building', 'street', 'wall', 'floor', 'ground', 'ceiling']:
            # For large architectural elements: use containment heuristic
            for obj in normalized_objects:
                if obj_class == obj["class"]:
                    # Try containment first (more lenient for partial detections)
                    if self.is_contained_or_contains(bbox_xyxy, obj["bbox_xyxy"], threshold=0.4):
                        return obj["object_id"]

                    # Also try IoU with low threshold
                    iou = self.calculate_iou(bbox_xyxy, obj["bbox_xyxy"])
                    if iou >= 0.3:
                        return obj["object_id"]

        elif obj_class in ['window', 'door', 'sign', 'light']:
            # For architectural details: consider area similarity
            area_ratio_threshold = 2.5

            for obj in normalized_objects:
                if obj_class == obj["class"]:
                    obj_area = obj["w"] * obj["h"]
                    area_ratio = max(bbox_area, obj_area) / min(bbox_area, obj_area)

                    # Only compare similar-sized objects
                    if area_ratio > area_ratio_threshold:
                        continue
                    
                    iou = self.calculate_iou(bbox_xyxy, obj["bbox_xyxy"])

                    # Use lenient threshold for similar sizes
                    effective_threshold = 0.5 * (2.5 / area_ratio)
                    if iou >= min(effective_threshold, 0.7):
                        return obj["object_id"]

        else:  # Regular objects: car, person, tire, etc.
            # Use standard IoU
            for obj in normalized_objects:
                if obj_class == obj["class"]:
                    iou = self.calculate_iou(bbox_xyxy, obj["bbox_xyxy"])

                    if iou >= 0.5:
                        return obj["object_id"]

        return None

    def merge_similar_objects(self, scene_graph, iou_thresh=0.4):
        """
        Post-process the scene graph to merge duplicate objects
        (e.g., same building detected multiple times with slightly different boxes).
        """
        objects = scene_graph["objects"]
        rels = scene_graph["relationships"]

        def xywh_to_xyxy(obj):
            return [obj["x"], obj["y"], obj["x"] + obj["w"], obj["y"] + obj["h"]]

        merged_map = {}
        merged = set()

        for i, obj1 in enumerate(objects):
            if obj1["object_id"] in merged:
                continue
            box1 = xywh_to_xyxy(obj1)
            for j, obj2 in enumerate(objects[i + 1:], i + 1):
                if obj2["object_id"] in merged:
                    continue
                if obj1["names"][0] != obj2["names"][0]:
                    continue

                box2 = xywh_to_xyxy(obj2)
                iou = self.calculate_iou(box1, box2)

                # More lenient merging for large static structures
                is_large_class = obj1["names"][0] in ["building", "street", "sky", "wall", "ground"]
                threshold = 0.3 if is_large_class else iou_thresh

                if iou >= threshold or self.is_contained_or_contains(box1, box2, 0.6):
                    merged.add(obj2["object_id"])
                    merged_map[obj2["object_id"]] = obj1["object_id"]

                    # Expand obj1 to cover both boxes
                    x1 = min(box1[0], box2[0])
                    y1 = min(box1[1], box2[1])
                    x2 = max(box1[2], box2[2])
                    y2 = max(box1[3], box2[3])
                    obj1["x"], obj1["y"] = x1, y1
                    obj1["w"], obj1["h"] = (x2 - x1), (y2 - y1)

                    # Merge attributes
                    for attr in obj2.get("attributes", []):
                        if attr not in obj1["attributes"]:
                            obj1["attributes"].append(attr)

        # Remove merged objects
        scene_graph["objects"] = [o for o in objects if o["object_id"] not in merged]

        # Update relationships to new IDs
        for r in rels:
            if r["subject_id"] in merged_map:
                r["subject_id"] = merged_map[r["subject_id"]]
            if r["object_id"] in merged_map:
                r["object_id"] = merged_map[r["object_id"]]

        # Deduplicate relationships
        unique_rels = []
        seen = set()
        for r in rels:
            key = (r["subject_id"], r["predicate"], r["object_id"])
            if key not in seen:
                seen.add(key)
                unique_rels.append(r)
        scene_graph["relationships"] = unique_rels

        print(f"✅ Merged {len(merged)} duplicate objects.")
        scene_graph = self.merge_similar_objects(scene_graph)
        return scene_graph


    def merge_object_attributes(self, existing_obj, new_attributes):
        """
        Merge attributes from multiple detections of the same object.
        Avoids duplicates.
        """
        for attr in new_attributes:
            if attr not in existing_obj["attributes"]:
                existing_obj["attributes"].append(attr)

        
    def build_scene_graph(self, input_img, image_name=None):
        if os.path.exists(f"data/{image_name}_scene_graph.json"):
            print("Scene graph already built. Returning existing graph.")
            with open(f"data/{image_name}_scene_graph.json", "r") as f:
                sg = json.load(f)
            return sg
        
        # Handle both PIL Image and tensor inputs
        if isinstance(input_img, torch.Tensor):
            # If it's already a tensor, store original size and assign
            if input_img.dim() == 3:
                self.original_size = (input_img.shape[2], input_img.shape[1])  # (width, height)
                self.img = input_img.unsqueeze(0)  # Add batch dimension
            elif input_img.dim() == 4:
                self.original_size = (input_img.shape[3], input_img.shape[2])  # (width, height)
                self.img = input_img
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {input_img.dim()}D")
        else:
            # If it's a PIL Image or similar, transform it
            self.img = input_img
            self.original_size = self.img.size  # (width, height)
            transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.img = transform(self.img).unsqueeze(0)

        position_embedding = PositionEmbeddingSine(128, normalize=True)
        backbone = Backbone('resnet50', False, False, False)
        backbone = Joiner(backbone, position_embedding)
        backbone.num_channels = 2048

        transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                                dim_feedforward=2048,
                                num_encoder_layers=6,
                                num_decoder_layers=6,
                                normalize_before=False,
                                return_intermediate_dec=True)
        
        self.model = RelTR(backbone, transformer, num_classes=151, num_rel_classes=51,
                        num_entities=100, num_triplets=200)

        # Load pretrained weights
        output_path = './reltr/checkpoint0149.pth'

        ckpt = torch.load(output_path, map_location='cpu', weights_only=False)
        print(ckpt.keys())
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        # Set up hooks to capture features
        conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
        hooks = [
            self.model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            self.model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                lambda self, input, output: dec_attn_weights_sub.append(output[1])
            ),
            self.model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                lambda self, input, output: dec_attn_weights_obj.append(output[1])
            )
        ]

        # Process the image
        self.outputs = self.model(self.img)

        # Store the captured features
        self.conv_features = conv_features[0]
        self.dec_attn_weights_sub = dec_attn_weights_sub[0]
        self.dec_attn_weights_obj = dec_attn_weights_obj[0]

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # keep only predictions with >0.3 confidence
        probas = self.outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = self.outputs['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = self.outputs['obj_logits'].softmax(-1)[0, :, :-1]
        self.keep = torch.logical_and(probas.max(-1).values > 0.3,
                                    torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                    probas_obj.max(-1).values > 0.3))
        
        topk = 30  # display up to 30 images
        self.keep_queries = torch.nonzero(self.keep, as_tuple=True)[0]
        self.indices = torch.argsort(-probas[self.keep_queries].max(-1)[0] * 
                                   probas_sub[self.keep_queries].max(-1)[0] * 
                                   probas_obj[self.keep_queries].max(-1)[0])[:topk]
        self.keep_queries = self.keep_queries[self.indices]
        self.probas = probas
        self.probas_sub = probas_sub
        self.probas_obj = probas_obj

        # Initialize Visual Genome format scene graph
        # Initialize Visual Genome format scene graph
        objects = []
        relationships = []
        normalized_objects = []  # Track objects in normalized space
        next_object_id = 1
        next_relationship_id = 1
        merge_stats = {}

        # Get NORMALIZED bounding boxes (don't rescale yet!)
        sub_bboxes_norm = self.outputs['sub_boxes'][0, self.keep]  # [N, 4] in [cx, cy, w, h] format
        obj_bboxes_norm = self.outputs['obj_boxes'][0, self.keep]  # [N, 4] in [cx, cy, w, h] format

        # Helper function to convert bbox format for final output
        def bbox_to_xywh(bbox):
            x1, y1, x2, y2 = bbox
            return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

        # Ensure there are predictions to process
        if len(self.keep_queries) > 0:
            topk = 30
            current_indices = self.indices[:min(topk, len(self.keep_queries))]

            for idx in current_indices:
                original_idx = self.keep_queries[idx]

                # Get the predicted classes and relationship
                relationship = self.REL_CLASSES[self.probas[original_idx].argmax()]
                subject_class = self.CLASSES[self.probas_sub[original_idx].argmax()]
                object_class = self.CLASSES[self.probas_obj[original_idx].argmax()]

                # Get the NORMALIZED bounding boxes (still in [cx, cy, w, h] format)
                bbox_index = current_indices.tolist().index(idx)
                subject_bbox_norm = sub_bboxes_norm[bbox_index].detach().cpu().tolist()
                object_bbox_norm = obj_bboxes_norm[bbox_index].detach().cpu().tolist()

                # ===== MATCH IN NORMALIZED SPACE (before rescaling) =====
                
                # Process subject
                subject_id = self.find_matching_object(subject_class, subject_bbox_norm, normalized_objects)
                
                if subject_id is None:
                    # Create new subject object
                    # First convert normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
                    subject_bbox_tensor = torch.tensor([subject_bbox_norm])
                    subject_bbox_scaled = self.rescale_bboxes(subject_bbox_tensor, self.original_size)[0].tolist()
                    
                    x, y, w, h = bbox_to_xywh(subject_bbox_scaled)
                    
                    # Add to objects list
                    objects.append({
                        "object_id": next_object_id,
                        "names": [subject_class],
                        "synsets": [],
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "attributes": []
                    })
                    
                    # Track in normalized space for future matching
                    cx, cy, bw, bh = subject_bbox_norm
                    normalized_objects.append({
                        "object_id": next_object_id,
                        "class": subject_class,
                        "bbox_xyxy": [cx - 0.5*bw, cy - 0.5*bh, cx + 0.5*bw, cy + 0.5*bh],
                        "w": bw,
                        "h": bh
                    })
                    
                    subject_id = next_object_id
                    next_object_id += 1
                else:
                    merge_stats[subject_class] = merge_stats.get(subject_class, 0) + 1

                # Process object
                object_id = self.find_matching_object(object_class, object_bbox_norm, normalized_objects)
                
                if object_id is None:
                    # Create new object
                    object_bbox_tensor = torch.tensor([object_bbox_norm])
                    object_bbox_scaled = self.rescale_bboxes(object_bbox_tensor, self.original_size)[0].tolist()
                    
                    x, y, w, h = bbox_to_xywh(object_bbox_scaled)
                    
                    objects.append({
                        "object_id": next_object_id,
                        "names": [object_class],
                        "synsets": [],
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "attributes": []
                    })
                    
                    # Track in normalized space
                    cx, cy, bw, bh = object_bbox_norm
                    normalized_objects.append({
                        "object_id": next_object_id,
                        "class": object_class,
                        "bbox_xyxy": [cx - 0.5*bw, cy - 0.5*bh, cx + 0.5*bw, cy + 0.5*bh],
                        "w": bw,
                        "h": bh
                    })
                    
                    object_id = next_object_id
                    next_object_id += 1
                else:
                    merge_stats[object_class] = merge_stats.get(object_class, 0) + 1

                # Create relationship (avoid duplicates)
                relationship_exists = any(
                    rel["subject_id"] == subject_id and 
                    rel["object_id"] == object_id and 
                    rel["predicate"] == relationship
                    for rel in relationships
                )
                
                if not relationship_exists:
                    relationships.append({
                        "relationship_id": next_relationship_id,
                        "subject_id": subject_id,
                        "object_id": object_id,
                        "predicate": relationship,
                        "synsets": []
                    })
                    next_relationship_id += 1

        # Create final scene graph in Visual Genome format
        scene_graph = {
            "image_id": 1,
            "objects": objects,
            "relationships": relationships
        }

        # Print statistics
        print(f"\n{'='*60}")
        print(f"Scene graph created with {len(objects)} unique objects and {len(relationships)} relationships")
        print(f"{'='*60}")
        
        if merge_stats:
            print("\nObject merging statistics:")
            for obj_type, count in sorted(merge_stats.items(), key=lambda x: -x[1]):
                print(f"  {obj_type:12} : {count:2} duplicates merged")
        
        return scene_graph
    
    def clip_visualize_scene_graph(self, scene_graph, image_name=None):
        """Extract visual attributes for objects in the scene graph using CLIP."""
        if os.path.exists(f"data/{image_name}_scene_graph_with_attributes.json"):
            print("Attributes already extracted.")
            with open(f"data/{image_name}_scene_graph_with_attributes.json", "r") as f:
                sg = json.load(f)
            return sg

        TOP_K = 5  # number of top attributes per object
        SIM_THRESHOLD = 0.18  # discard weak matches
        EMBEDDINGS_FILE = "C:\\Users\\SANJIV\\OneDrive\\Desktop\\PES\\SemVI\\Capstone\\SceneDesc\\urban-scene-intelligence\\clip_attribute_embeddings.pt"
        MODEL = "C:\\Users\\SANJIV\\OneDrive\\Desktop\\PES\\SemVI\\Capstone\\SceneDesc\\urban-scene-intelligence\\clip_model"

        if not os.path.exists(EMBEDDINGS_FILE):
            raise FileNotFoundError(f"CLIP embeddings file not found: {EMBEDDINGS_FILE}")

        # Loading the CLIP model
        model = CLIPModel.from_pretrained(MODEL).to(self.device)
        processor = CLIPProcessor.from_pretrained(MODEL)
        model.eval()

        # Convert tensor to PIL Image for cropping
        with torch.no_grad():
            # Get the normalized image tensor
            img_tensor = self.img.squeeze(0)  # Remove batch dimension [C, H, W]
            
            # Create broadcasting-compatible mean and std tensors
            mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device)
            
            # Reshape mean and std for broadcasting
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
            
            # Denormalize the image
            img_tensor = img_tensor * std + mean
            img_tensor = img_tensor.clamp(0, 1)
            
            # Convert to PIL Image format
            img_tensor = img_tensor.cpu()
            image_pil = T.ToPILImage()(img_tensor)
        
        # Load the saved data
        loaded_data = torch.load(EMBEDDINGS_FILE)

        # Extract embeddings and vocab
        text_embeds = loaded_data["embeddings"]
        attribute_vocab = loaded_data["vocab"]

        for obj in tqdm(scene_graph.get("objects", []), desc="Processing objects"):
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            crop = image_pil.crop((int(x), int(y), int(x + w), int(y + h)))
            
            image_inputs = processor(images=crop, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_embed = model.get_image_features(**image_inputs)
                image_embed /= image_embed.norm(dim=-1, keepdim=True)
                sims = (image_embed @ text_embeds.T.to(self.device)).squeeze(0).detach().cpu().numpy()
                
            del image_inputs, image_embed
            torch.cuda.empty_cache()

            top_indices = np.argsort(sims)[::-1][:TOP_K]
            top_attrs = [attribute_vocab[i] for i in top_indices if sims[i] > SIM_THRESHOLD]
            obj["attributes"] = top_attrs

            gc.collect()
        
        with open(f"data/{image_name}_scene_graph_with_attributes.json", "w") as f:
            json.dump(scene_graph, f, indent=4)
        return scene_graph

    def visualize_scene_graph(self, image_name=None):
        """
        Visualize the scene graph with attention maps and bounding boxes.
        Args:
            graph: The scene graph to visualize
            image_name: Optional name of the input image for caching
        """
        # Use the stored features instead of extracting them again
        if self.conv_features is None or self.dec_attn_weights_sub is None or self.dec_attn_weights_obj is None:
            print("Warning: Features not available. Please process an image first.")
            return
            
        # Create visualizations directory if it doesn't exist
        os.makedirs(os.path.join('data', 'visualizations'), exist_ok=True)

        # get the feature map shape and transformed image size
        h, w = self.conv_features['0'].tensors.shape[-2:]
        img_h, img_w = self.img.shape[2:]  # Get the height and width of the transformed image
        
        # Scale boxes to the transformed image size
        sub_bboxes_scaled = self.rescale_bboxes(self.outputs['sub_boxes'][0, self.keep], (img_w, img_h))
        obj_bboxes_scaled = self.rescale_bboxes(self.outputs['obj_boxes'][0, self.keep], (img_w, img_h))

        # Create a figure with proper size based on number of predictions
        n_preds = len(self.indices)
        if n_preds == 0:
            print("No predictions to visualize")
            return
            
        fig, axs = plt.subplots(ncols=n_preds, nrows=3, figsize=(22, 7))
        
        # Handle the case where there's only one prediction
        if n_preds == 1:
            axs = axs.reshape(-1, 1)  # Reshape to make it 2D

        # Convert bounding boxes to numpy arrays
        sub_boxes_np = sub_bboxes_scaled[self.indices].detach().cpu().numpy()
        obj_boxes_np = obj_bboxes_scaled[self.indices].detach().cpu().numpy()
        
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(self.keep_queries, axs.T, sub_boxes_np, obj_boxes_np):
            # Subject attention visualization
            ax = ax_i[0]
            attention_map_sub = self.dec_attn_weights_sub[0, idx].view(h, w).detach().cpu().numpy()
            ax.imshow(attention_map_sub)
            ax.axis('off')
            ax.set_title(f'Subject Attention {idx.item()}')

            # Object attention visualization
            ax = ax_i[1]
            attention_map_obj = self.dec_attn_weights_obj[0, idx].view(h, w).detach().cpu().numpy()
            ax.imshow(attention_map_obj)
            ax.axis('off')
            ax.set_title(f'Object Attention {idx.item()}')

            # Original image with bounding boxes
            ax = ax_i[2]
            img_np = self.img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            # Denormalize the image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np)
            ax.add_patch(plt.Rectangle((float(sxmin), float(symin)), 
                                     float(sxmax - sxmin), float(symax - symin),
                                     fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((float(oxmin), float(oymin)), 
                                     float(oxmax - oxmin), float(oymax - oymin),
                                     fill=False, color='orange', linewidth=2.5))

            ax.axis('off')
            ax.set_title(f'{self.CLASSES[self.probas_sub[idx].argmax()]}\n{self.REL_CLASSES[self.probas[idx].argmax()]}\n{self.CLASSES[self.probas_obj[idx].argmax()]}', 
                        fontsize=8)

        plt.suptitle("Blue: Subject, Orange: Object", y=1.02)
        fig.tight_layout()
        
        # Save the visualization if image_name is provided
        if image_name:
            vis_path = os.path.join('data', 'visualizations', f"{image_name}_vis.png")
            fig.savefig(vis_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
        return fig  # Return the figure instead of showing it directly


    def save_scene_graph(self, scene_graph, image_name=None):
        output_path = f"data/{image_name}_scene_graph.json"
        with open(output_path, "w") as f:
            json.dump(scene_graph, f, indent=4)
        print(f"✅ Scene graph saved to {output_path}")


if __name__ == "__main__":
    generator = SceneGraphGenerator()
    # You need to provide an input image
    input_img = Image.open('path_to_your_image.jpg')  # Replace with your image path
    graph = generator.build_scene_graph(input_img)
    generator.save_scene_graph(graph)
    generator.visualize_scene_graph(graph)  # Optional: visualize the scene graph

