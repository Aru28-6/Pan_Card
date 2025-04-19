import streamlit as st
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Preprocess image (resize & grayscale)
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((300, 150))                # Resize to smaller dimension
    return np.array(image)

# Compare images using SSIM
def compare_images(image1, image2):
    score, diff = ssim(image1, image2, full=True)
    return score, (diff * 255).astype("uint8")

# Decide result based on SSIM
def determine_result(score):
    return ("✅ Valid PAN Card", "Minimal structural differences.") if score >= 0.85 else ("❌ Fake PAN Card", "Major structural differences detected.")

# Streamlit UI
st.title("🆔 PAN Card Tampering Detection")

ref_image_file = st.file_uploader("Upload reference PAN card image (original)", type=["png", "jpg", "jpeg"])
test_image_file = st.file_uploader("Upload test PAN card image (to be checked)", type=["png", "jpg", "jpeg"])

if ref_image_file and test_image_file:
    ref_image = preprocess_image(ref_image_file)
    test_image = preprocess_image(test_image_file)

    score, diff = compare_images(ref_image, test_image)
    result, reason = determine_result(score)

    st.subheader("📊 SSIM Comparison Result")
    st.write(f"**SSIM Score:** `{score:.4f}`")
    st.write(f"**Result:** {result}")
    st.write(f"**Reason:** {reason}")

    st.subheader("🖼 Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(ref_image_file, caption="Reference Image", use_column_width=True)
    with col2:
        st.image(test_image_file, caption="Test Image", use_column_width=True)

    st.subheader("📌 Structural Difference Map")
    fig, ax = plt.subplots()
    ax.imshow(diff, cmap='hot')
    ax.axis('off')
    st.pyplot(fig)
