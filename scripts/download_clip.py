"""
Script to download and save the CLIP model and processor locally.
This should be run once before using the scene understanding pipeline.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
import os

def download_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Using device: {device}")

    # Define local path relative to project root
    LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clip_model")
    
    # Ensure the directory exists
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    print(f"🔹 Downloading CLIP model to: {LOCAL_MODEL_DIR}")
    
    try:
        # Load from Hugging Face and save locally
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Save model and processor
        model.save_pretrained(LOCAL_MODEL_DIR)
        processor.save_pretrained(LOCAL_MODEL_DIR)
        
        print("✅ CLIP model and processor saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading CLIP model: {str(e)}")
        return False

if __name__ == "__main__":
    download_clip_model()