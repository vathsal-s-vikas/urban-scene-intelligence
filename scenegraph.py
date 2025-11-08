# stage2_scene_graph_generator.py
import json
import numpy as np
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import requests
import matplotlib.pyplot as plt
from reltr.models.backbone import Backbone, Joiner
from reltr.models.position_encoding import PositionEmbeddingSine
from reltr.models.transformer import Transformer
from reltr.models.reltr import RelTR
import os
import torch # Import torch here

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

    def build_scene_graph(self, input_img):
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

        # Check if the downloaded file size is reasonable (e.g., greater than 1MB)
        # You might need to adjust this threshold based on the expected size of the checkpoint file
        expected_size_threshold = 1024 * 1024  # 1MB
        if not os.path.exists(output_path) or os.path.getsize(output_path) < expected_size_threshold:
            print(f"Downloaded file {output_path} seems too small. Trying an alternative download method.")

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

        # Generate scene graph
        scene_graph = []

        # Get scaled bounding boxes using original image size
        sub_bboxes_scaled = self.rescale_bboxes(self.outputs['sub_boxes'][0, self.keep], self.original_size)
        obj_bboxes_scaled = self.rescale_bboxes(self.outputs['obj_boxes'][0, self.keep], self.original_size)

        # Ensure there are predictions to process
        if len(self.keep_queries) > 0:
            # Limit indices to the number of available queries if topk is larger
            topk = 30  # display up to 30 images
            current_indices = self.indices[:min(topk, len(self.keep_queries))]

            for idx in current_indices:  # Iterate through the indices of the topk predictions in the filtered list
                # Get the original index in the filtered list 'keep_queries'
                original_idx = self.keep_queries[idx]

                # Get the predicted relationship, subject, and object classes using the original index
                relationship = self.REL_CLASSES[self.probas[original_idx].argmax()]
                subject_class = self.CLASSES[self.probas_sub[original_idx].argmax()]
                object_class = self.CLASSES[self.probas_obj[original_idx].argmax()]

                # Get the scaled bounding boxes for the subject and object
                bbox_index = current_indices.tolist().index(idx)  # Find the position of 'idx' within 'current_indices'
                sxmin, symin, sxmax, symax = sub_bboxes_scaled[bbox_index]
                oxmin, oymin, oxmax, oymax = obj_bboxes_scaled[bbox_index]

                # Create a dictionary for the triplet (subject, relationship, object)
                triplet = {
                    'subject': {'class': subject_class, 'bbox': [sxmin.item(), symin.item(), sxmax.item(), symax.item()]},
                    'relationship': relationship,
                    'object': {'class': object_class, 'bbox': [oxmin.item(), oymin.item(), oxmax.item(), oymax.item()]}
                }
                scene_graph.append(triplet)

        return scene_graph

    def visualize_scene_graph(self, graph):
        # Use the stored features instead of extracting them again
        if self.conv_features is None or self.dec_attn_weights_sub is None or self.dec_attn_weights_obj is None:
            print("Warning: Features not available. Please process an image first.")
            return

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
        return fig  # Return the figure instead of showing it directly


    def save_scene_graph(self, scene_graph, output_path="data/scene_graph.json"):
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
