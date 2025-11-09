import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scenegraph import semantic_graph_to_text
import scenegraph as sg

# ===============================
# Load Flan-T5
# ===============================
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# ===============================
# Refinement function
# ===============================
def refine_description(G, current_caption, user_prompt):
    """
    Refines the caption using semantic graph + user instruction.
    """
    scene_graph_text = semantic_graph_to_text(G)
    input_text = (
        f"Scene: {scene_graph_text}\n"
        f"Current caption: {current_caption}\n"
        f"Refine caption: {user_prompt}"
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    refined_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return refined_caption

# ===============================
# Main interactive loop
# ===============================
if __name__ == "__main__":
    # Stage 2: Generate semantic scene graph
    image_path = "data/sample_images/street.jpg"
    G = sg.generate_scene_graph_semantic(image_path)

    # Stage 3: Initial caption (you can also read from file)
    original_caption = "car next to person"
    refined_caption = original_caption

    print("\n--- Stage 4: Interactive Refinement ---")
    print(f"Initial caption: {refined_caption}")

    # Up to 3 rounds of user refinement
    for round_idx in range(3):
        user_prompt = input(f"\nRound {round_idx+1} - Enter refinement request (or 'skip'): ").strip()
        if user_prompt.lower() == "skip" or user_prompt == "":
            break
        refined_caption = refine_description(G, refined_caption, user_prompt)
        print(f"Refined caption after round {round_idx+1}: {refined_caption}")

    # Save final refined caption
    os.makedirs("data", exist_ok=True)
    output_path = "data/narrative_description.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(refined_caption)
    
    print(f"\n✅ Final refined description saved to {output_path}")
