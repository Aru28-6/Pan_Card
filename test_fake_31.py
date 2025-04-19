import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Preprocess image (resize & grayscale)
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((400, 200))
    return np.array(image)

# SSIM comparison
def compare_images(image1, image2):
    score, diff = ssim(image1, image2, full=True)
    return score, (diff * 255).astype("uint8")

# Decide result
def determine_result(score):
    return ("✅ Valid PAN Card", "Minimal structural differences.") if score >= 0.85 else ("❌ Fake PAN Card", "Major structural differences detected.")

# Streamlit UI
st.set_page_config(page_title="PAN Card Tampering Detection", layout="centered")
st.title("🆔 PAN Card Tampering Detection using SSIM")

# Upload images
ref_image_file = st.file_uploader("📌 Upload *reference* (original) PAN card", type=["png", "jpg", "jpeg"])
test_image_file = st.file_uploader("🔍 Upload *test* PAN card image to verify", type=["png", "jpg", "jpeg"])

if ref_image_file and test_image_file:
    # Process both images
    ref_image = preprocess_image(ref_image_file)
    test_image = preprocess_image(test_image_file)

    # Compare using SSIM
    score, diff = compare_images(ref_image, test_image)
    result, reason = determine_result(score)

    # Display result
    st.subheader("📊 SSIM Result")
    st.metric(label="SSIM Score", value=f"{score:.4f}")
    st.write(f"**Result:** {result}")
    st.info(reason)

    # Show uploaded images
    st.subheader("🖼 Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(ref_image_file, caption="Reference PAN Card", use_column_width=True)
    with col2:
        st.image(test_image_file, caption="Test PAN Card", use_column_width=True)

    # Show SSIM Difference Map
    st.subheader("📌 Structural Difference Map (SSIM Heatmap)")
    fig, ax = plt.subplots()
    ax.imshow(diff, cmap='hot')
    ax.axis("off")
    st.pyplot(fig)
