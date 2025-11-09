# app.py
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from scenegraph import SceneGraphGenerator
from description_generator import generate_description_from_entry

# Configure Streamlit page
st.set_page_config(page_title="Urban Scene Intelligence", layout="wide")

# Project Description
st.title("🏙️ Urban Scene Intelligence")
st.markdown("""
### About this Project
This application analyzes urban scenes using advanced computer vision and natural language processing:

1. **Scene Analysis**: Detects objects, their relationships, and spatial arrangements in urban images
2. **Relationship Detection**: Identifies how objects interact with each other (e.g., "car parked on street")
3. **Scene Graph Generation**: Creates a structured representation of the scene
4. **Natural Description**: Generates human-readable descriptions of the urban environment

Upload an urban scene image to get started!
""")

# Sidebar for image upload
st.sidebar.header("Upload Urban Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Initialize session state for the generator if not exists
if 'scene_generator' not in st.session_state:
    st.session_state['scene_generator'] = SceneGraphGenerator()

# Main application logic
if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Urban Scene", use_column_width=True)
    
    # Process the image
    st.subheader("1️⃣ Scene Analysis")
    with st.spinner("Analyzing the scene..."):
        generator = st.session_state['scene_generator']
        scene_graph = generator.build_scene_graph(img)
        complete_scene_graph = generator.clip_visualize_scene_graph()
        
    # Option to visualize RELTR attention
    st.subheader("2️⃣ Relationship Detection Visualization")
    if st.button("Show Relationship Detection Visualization"):
        with st.spinner("Generating visualization..."):
            generator.visualize_scene_graph(scene_graph)
            st.pyplot(plt.gcf())
            plt.close()
    
    # Display scene graph
    st.subheader("3️⃣ Scene Graph")
    with st.expander("View Generated Scene Graph", expanded=False):
        st.json(complete_scene_graph)
    
    # Generate and display description
    st.subheader("4️⃣ Scene Description")
    if st.button("Generate Description"):
        with st.spinner("Generating natural description..."):
            description = generate_description_from_entry(scene_graph)
            st.text_area("📝 Generated Description", value=description, height=200)

    # Save results
    os.makedirs("data", exist_ok=True)
    generator.save_scene_graph(scene_graph)
    
    # Comment out refinement section for now
    """
    # ---- Optional Refinement (To be implemented) ----
    st.subheader("5️⃣ Description Refinement")
    clarification = st.text_input("Refine the description (e.g., focus on traffic or pedestrians):", value="")
    if st.button("Refine Description") and clarification.strip():
        with st.spinner("Refining description..."):
            refined_desc = refine_description(scene_graph, description, clarification.strip())
        st.text_area("📝 Refined Description", value=refined_desc, height=200)
    """
    
else:
    # Show example/demo information when no image is uploaded
    st.info("👆 Upload an urban scene image using the sidebar to begin analysis")
    st.markdown("""
    ### What to expect:
    1. Upload any urban scene image (streets, buildings, traffic, etc.)
    2. View the attention visualization showing how the AI understands object relationships
    3. Examine the structured scene graph representation
    4. Read a natural language description of the scene
    """)
