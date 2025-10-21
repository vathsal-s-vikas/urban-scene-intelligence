# description_generator.py

import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_scene_graph(json_path):
    """Load scene graph from JSON"""
    with open(json_path, "r") as f:
        graph = json.load(f)
    return graph

def graph_to_sentences(graph):
    """
    Convert structured scene graph into descriptive statements.
    Returns a list of sentences like "car is left of building".
    """
    sentences = []

    # Nodes dictionary for easy lookup
    node_lookup = {node['id']: node for node in graph['nodes']}

    # Convert edges to sentences
    for edge in graph['edges']:
        source = node_lookup[edge['source']]
        target = node_lookup[edge['target']]
        relation = edge.get('relation', 'near')
        
        # Include attributes if any
        source_attrs = ", ".join(source.get("attributes", []))
        target_attrs = ", ".join(target.get("attributes", []))
        
        subj = f"{source_attrs + ' ' if source_attrs else ''}{source['label']}"
        obj = f"{target_attrs + ' ' if target_attrs else ''}{target['label']}"
        
        sentences.append(f"{subj} {relation} {obj}")
    
    # Include scene attributes as a single sentence
    scene_attrs = graph.get('scene_attributes', {})
    scene_desc = []
    if 'time_of_day' in scene_attrs:
        scene_desc.append(scene_attrs['time_of_day'])
    if 'weather' in scene_attrs:
        scene_desc.append(scene_attrs['weather'])
    if scene_desc:
        sentences.append(f"The scene is {', '.join(scene_desc)}.")

    return sentences

def generate_narrative_description(graph, model_name="google/flan-t5-base", max_length=300, temperature=0.7):
    """
    Generate coherent urban description from structured scene graph.
    """
    sentences = graph_to_sentences(graph)
    if not sentences:
        return "No significant urban objects detected."

    # Rich prompt for context-aware description
    narrative_input = (
        "You are an expert urban observer analyzing a city street scene. "
        "Generate a fluent paragraph describing the environment, activities, and spatial layout. "
        "Focus on traffic, buildings, pedestrians, and overall ambience.\n\n"
        "Scene facts:\n"
        + "; ".join(sentences)
        + "\n\nProvide a cohesive narrative."
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(narrative_input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            temperature=temperature,
            early_stopping=True,
            do_sample=True
        )

    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description.strip()

if __name__ == "__main__":
    # Example usage
    from stage2_semantic_scene_graph import build_scene_graph

    # Dummy detections
    detections = [
        {"label": "car", "bbox": [100, 300, 220, 400], "confidence": 0.89},
        {"label": "building", "bbox": [20, 100, 280, 600], "confidence": 0.95},
        {"label": "tree", "bbox": [300, 250, 370, 420], "confidence": 0.88}
    ]

    graph = build_scene_graph(detections, weather="sunny", time_of_day="daytime")
    narrative = generate_narrative_description(graph)
    print("\n📝 Generated Urban Scene Description:\n")
    print(narrative)
