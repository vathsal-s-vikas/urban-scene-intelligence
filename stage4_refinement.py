from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from stage2_semantic_scene_graph import semantic_graph_to_text
import networkx as nx
import pickle  # optional, if you save/load graphs

# Load Flan-T5
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def refine_description(G, original_caption, user_prompt):
    """
    G: semantic scene graph (nx.DiGraph)
    original_caption: text from Stage 3
    user_prompt: user instruction for refinement
    """
    scene_graph_text = semantic_graph_to_text(G)
    input_text = f"Scene: {scene_graph_text}\nOriginal caption: {original_caption}\nRefine caption: {user_prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    refined_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return refined_caption

# Example usage
if __name__ == "__main__":
    # Load graph from Stage 2
    import stage2_semantic_scene_graph as sg
    G = sg.generate_scene_graph_semantic("data/sample_images/street.jpg")
    
    original_caption = "car next to person"
    user_prompt = "Include the traffic light and weather"
    
    refined_caption = refine_description(G, original_caption, user_prompt)
    print("📝 Refined Description:", refined_caption)
