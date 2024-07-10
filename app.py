import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Function to extract dominant colors
def extract_dominant_colors(image, num_colors):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((100, 100))
    img_array = np.array(image)
    img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img_array)
    colors = kmeans.cluster_centers_

    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    dominant_colors = colors[label_counts.argsort()[::-1]]

    return dominant_colors.astype(int), kmeans

# Function to recolor image
def recolor_image(image, kmeans):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)
    img_array = img_array.reshape((-1, 3))

    labels = kmeans.predict(img_array)
    recolored_img_array = kmeans.cluster_centers_[labels]
    recolored_img_array = recolored_img_array.reshape((image.height, image.width, 3)).astype(np.uint8)

    recolored_image = Image.fromarray(recolored_img_array)
    return recolored_image

# Function to create dominant color image
def create_dominant_color_image(dominant_colors, image_shape):
    color_blocks = np.zeros(image_shape, dtype=np.uint8)
    block_height = image_shape[0] // len(dominant_colors)
    remainder_height = image_shape[0] % len(dominant_colors)

    start_height = 0
    for i, color in enumerate(dominant_colors):
        end_height = start_height + block_height + (1 if i < remainder_height else 0)
        color_blocks[start_height:end_height, :] = color
        start_height = end_height

    return Image.fromarray(color_blocks)

# Streamlit app
st.title("Dominant Color Extraction")

# Session state to manage navigation
if 'page' not in st.session_state:
    st.session_state.page = 'homepage'

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Homepage", "Upload Image"])

# Homepage content
if page == 'Homepage':
    st.header("Welcome to the Dominant Color Extraction App")
    st.write("""
        This app allows you to upload an image and extract the dominant colors from it. 
        You can also see the recolored version of your image using these dominant colors.
        
        ### How to use this application:
        1. Select 'Upload Image' from the navigation bar.
        2. Upload your image file (jpg, jpeg, or png).
        3. Choose the number of dominant colors you want to extract.
        4. View the original image, the recolored image, and the dominant colors.
    """)

# Upload page content
elif page == 'Upload Image':
    st.header("Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        num_colors = st.slider("Number of dominant colors", min_value=2, max_value=100, value=5)
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Original Image', use_column_width=True)
        
        dominant_colors, kmeans = extract_dominant_colors(image, num_colors)
        
        recolored_img = recolor_image(image, kmeans)
        
        with col2:
            st.image(recolored_img, caption='Recolored Image with Dominant Colors', use_column_width=True)
        
        st.write("Top Dominant Colors:")
        dominant_color_image = create_dominant_color_image(dominant_colors, (100, 100, 3))
        st.image(dominant_color_image, caption='Dominant Colors Image', use_column_width=True)

        for i, color in enumerate(dominant_colors):
            st.write(f"Color {i+1}: RGB({color[0]}, {color[1]}, {color[2]})")
