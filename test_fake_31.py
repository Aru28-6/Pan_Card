import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Preprocess uploaded images
def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (400, 250))
    return image

# Compare two images using SSIM
def compare_images(image1, image2):
    score, diff = ssim(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

# Highlight tampered areas and extract contours
def highlight_tampered_sections(reference, test, diff):
    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    tampered_areas = []

    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            tampered_areas.append((x, y, w, h))

    return result_image, thresh, tampered_areas

# Determine result from SSIM score
def determine_result(score):
    if score >= 0.85:
        return "✅ Valid PAN Card", "High similarity score and minimal structural differences."
    else:
        return "❌ Fake PAN Card", "Major structural differences detected. Possible tampering."

# Optional field regions (for label detection)
regions = {
    "PAN Number": (30, 10, 200, 70),
    "Candidate Name": (210, 10, 370, 70),
    "Father's/Mother's Name": (30, 80, 200, 130),
    "Date of Birth": (210, 80, 370, 130),
    "Photo": (30, 140, 160, 210),
    "QR Code": (260, 140, 390, 210),
    "Signature": (130, 210, 270, 250)
}

# Streamlit App UI
st.set_page_config(page_title="PAN Card Tampering Detection", layout="wide")
st.title("🔍 PAN Card Tampering Detection App")
st.write("Upload the original and one or more suspected PAN card images to check for tampering.")

# Upload original and test files
reference_file = st.file_uploader("Upload Original PAN Card", type=["png", "jpg", "jpeg"])
test_files = st.file_uploader("Upload Test PAN Cards", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Data for SSIM comparison graph
ssim_scores = []
file_names = []

# Begin processing if both original and test files are provided
if reference_file and test_files:
    ref_img = preprocess_image(reference_file)

    for idx, test_file in enumerate(test_files):
        test_img = preprocess_image(test_file)
        score, diff = compare_images(ref_img, test_img)
        result_img, thresh_img, tampered_areas = highlight_tampered_sections(ref_img, test_img, diff)
        result, reason = determine_result(score)

        ssim_scores.append(score)
        file_names.append(test_file.name)

        # Check which fields are tampered
        detected_sections = set()
        for (x, y, w, h) in tampered_areas:
            cx, cy = x + w // 2, y + h // 2
            for label, (x1, y1, x2, y2) in regions.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    detected_sections.add(f"❌ {label} is tampered.")

        # Show Results
        st.subheader(f"📄 Results for: {test_file.name}")
        col1, col2 = st.columns(2)
        with col1:
            st.image(ref_img, caption="Original PAN Card", channels="GRAY", use_container_width=True)
            st.image(diff, caption="SSIM Difference Image", channels="GRAY", use_container_width=True)
        with col2:
            st.image(thresh_img, caption="Thresholded Tampering Detection", channels="GRAY", use_container_width=True)
            st.image(result_img, caption="Detected Tampered Sections", use_container_width=True)

        st.write(f"**SSIM Score:** {score:.4f}")
        st.write(f"**Result:** {result}")
        st.write(f"**Reason:** {reason}")

        if detected_sections:
            st.write("### Tampered Fields:")
            for section in sorted(detected_sections):
                st.write(section)
        else:
            st.write("✅ No specific tampered fields detected.")
        st.markdown("---")

    # Plot SSIM scores across test images
    st.subheader("📊 SSIM Score Comparison with Original Image")

    # Custom labels for x-axis
    x_labels = [f"Test Image {i+1}" for i in range(len(file_names))]

    # Plotting
    fig, ax = plt.subplots()
    bars = ax.bar(x_labels, ssim_scores, color='skyblue', label='Test Images')
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Original Image (SSIM = 1.0)')

    ax.set_ylabel("Structural Similarity Index (SSIM)")
    ax.set_xlabel("Uploaded Test Images")
    ax.set_title("SSIM Comparison of Test Images vs Original")
    ax.set_ylim([0, 1.05])
    plt.xticks(rotation=45)

    for bar, score in zip(bars, ssim_scores):
        height = bar.get_height()
        ax.annotate(f"{score:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    ax.legend()
    st.pyplot(fig)
