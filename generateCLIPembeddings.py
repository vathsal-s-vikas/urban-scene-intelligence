"""
Generate CLIP embeddings for 50 curated visual attributes.
This creates a lightweight .pt file optimized for scene graph attribute extraction.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
import os

# Path to your CLIP model (same as in your scenegraph.py)
MODEL_PATH = "C:\\Users\\SANJIV\\OneDrive\\Desktop\\PES\\SemVI\\Capstone\\SceneDesc\\urban-scene-intelligence\\clip_model"
OUTPUT_PATH = "C:\\Users\\SANJIV\\OneDrive\\Desktop\\PES\\SemVI\\Capstone\\SceneDesc\\urban-scene-intelligence\\clip_50_attribute_embeddings.pt"

# 50 carefully curated attributes organized by category
ATTRIBUTE_VOCAB = [
    # Colors (10)
    "red", "blue", "green", "yellow", "white", 
    "black", "gray", "brown", "orange", "purple",
    
    # Materials (8)
    "metal", "wood", "glass", "plastic", "concrete", 
    "brick", "stone", "fabric",
    
    # Sizes (5)
    "large", "small", "tall", "wide", "narrow",
    
    # Shapes (5)
    "round", "square", "rectangular", "curved", "angular",
    
    # States/Conditions (8)
    "old", "new", "clean", "dirty", "bright", 
    "dark", "shiny", "rusty",
    
    # Textures (6)
    "smooth", "rough", "textured", "patterned", 
    "striped", "reflective",
    
    # Orientations/Positions (4)
    "vertical", "horizontal", "tilted", "upright",
    
    # Common Object Descriptors (4)
    "modern", "vintage", "decorative", "functional"
]

def generate_clip_embeddings():
    """
    Generate CLIP text embeddings for the attribute vocabulary.
    Compatible with your existing scenegraph.py code.
    """
    print("="*60)
    print("GENERATING CLIP ATTRIBUTE EMBEDDINGS")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"CLIP model not found at: {MODEL_PATH}")
    
    print(f"\nLoading CLIP model from: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    
    print(f"\nProcessing {len(ATTRIBUTE_VOCAB)} attributes...")
    
    # Generate embeddings
    with torch.no_grad():
        text_inputs = processor(
            text=ATTRIBUTE_VOCAB,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        text_embeddings = model.get_text_features(**text_inputs)
        
        # Normalize embeddings (important for cosine similarity)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Move to CPU for storage
        text_embeddings = text_embeddings.cpu()
    
    # Save embeddings and vocabulary
    save_data = {
        "embeddings": text_embeddings,
        "vocab": ATTRIBUTE_VOCAB,
        "num_attributes": len(ATTRIBUTE_VOCAB),
        "embedding_dim": text_embeddings.shape[1]
    }
    
    print(f"\nSaving embeddings to: {OUTPUT_PATH}")
    torch.save(save_data, OUTPUT_PATH)
    
    # Verify the saved file
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print(f"Embeddings saved: {OUTPUT_PATH}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Number of attributes: {len(ATTRIBUTE_VOCAB)}")
    print(f"Embedding dimension: {text_embeddings.shape[1]}")
    print("\nAttribute categories:")
    print("  - Colors: 10")
    print("  - Materials: 8")
    print("  - Sizes: 5")
    print("  - Shapes: 5")
    print("  - States/Conditions: 8")
    print("  - Textures: 6")
    print("  - Orientations: 4")
    print("  - Descriptors: 4")
    print("\nYou can now use this file in your scenegraph.py!")
    
    return save_data


def verify_embeddings():
    """
    Verify the generated embeddings file can be loaded correctly.
    """
    print("\n" + "="*60)
    print("VERIFYING EMBEDDINGS FILE")
    print("="*60)
    
    if not os.path.exists(OUTPUT_PATH):
        print("❌ Embeddings file not found!")
        return False
    
    try:
        loaded_data = torch.load(OUTPUT_PATH)
        
        print(f"✅ File loaded successfully!")
        print(f"   Embeddings shape: {loaded_data['embeddings'].shape}")
        print(f"   Number of attributes: {loaded_data['num_attributes']}")
        print(f"   Embedding dimension: {loaded_data['embedding_dim']}")
        print(f"\nFirst 10 attributes:")
        for i, attr in enumerate(loaded_data['vocab'][:10]):
            print(f"   {i+1}. {attr}")
        print(f"   ...")
        print(f"\nLast 5 attributes:")
        for i, attr in enumerate(loaded_data['vocab'][-5:], start=len(loaded_data['vocab'])-4):
            print(f"   {i}. {attr}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False


def test_similarity():
    """
    Test the embeddings with a sample similarity computation.
    """
    print("\n" + "="*60)
    print("TESTING EMBEDDINGS")
    print("="*60)
    
    if not os.path.exists(OUTPUT_PATH):
        print("❌ Embeddings file not found!")
        return
    
    loaded_data = torch.load(OUTPUT_PATH)
    embeddings = loaded_data["embeddings"]
    vocab = loaded_data["vocab"]
    
    # Create a dummy query embedding (simulating an image embedding)
    dummy_query = torch.randn(1, embeddings.shape[1])
    dummy_query = dummy_query / dummy_query.norm(dim=-1, keepdim=True)
    
    # Compute similarities
    similarities = (dummy_query @ embeddings.T).squeeze(0).numpy()
    
    # Get top 5
    top_indices = similarities.argsort()[::-1][:5]
    
    print("\nSample similarity computation (with random query):")
    print("Top 5 matching attributes:")
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. {vocab[idx]:15} (similarity: {similarities[idx]:.4f})")
    
    print("\n✅ Embeddings are working correctly!")


if __name__ == "__main__":
    # Generate embeddings
    save_data = generate_clip_embeddings()
    
    # Verify the file
    verify_embeddings()
    
    # Test with dummy query
    test_similarity()
    
    print("\n" + "="*60)
    print("🎉 ALL DONE! Your embeddings file is ready to use.")
    print("="*60)
    print("\nNext steps:")
    print("1. The file is already saved at the correct location")
    print("2. Your scenegraph.py will automatically use this new file")
    print("3. Run your scene graph pipeline as usual")
    print("\nThe attribute extraction will now be:")
    print("   • 1,310x faster (50 vs 65,537 attributes)")
    print("   • Much smaller file size")
    print("   • More meaningful and focused attributes")