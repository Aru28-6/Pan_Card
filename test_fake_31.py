import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Preprocess uploaded images
def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (400, 250))  # Resize image to a fixed size
    return image

# Compare two images using SSIM
def compare_images(image1, image2):
    score, diff = ssim(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")  # Scale difference to visible range
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

# Determine if the PAN card is fake or valid based on SSIM score
def determine_result(score):
    if score >= 0.85:
        return "Valid PAN Card", "✅ High similarity score and minimal structural differences."
    else:
        return "Fake PAN Card", "❌ Major structural differences detected. Possible fake document."

# Streamlit app UI
st.title("🔍 PAN Card Tampering Detection App")
st.write("Upload the original and suspected PAN card images.")

reference_file = st.file_uploader("Upload Original PAN Card", type=["png", "jpg", "jpeg"])
test_files = st.file_uploader("Upload Test PAN Cards", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if reference_file and test_files:
    # Preprocess the reference image (original PAN card)
    ref_img = preprocess_image(reference_file)

    # Loop through all the uploaded test files
    for test_file in test_files:
        # Preprocess the test image
        test_img = preprocess_image(test_file)
        # Compare images and get SSIM score and difference image
        score, diff = compare_images(ref_img, test_img)
        # Highlight tampered areas and extract contours
        result_img, thresh_img, tampered_areas = highlight_tampered_sections(ref_img, test_img, diff)

        # Determine result based on SSIM score
        result, reason = determine_result(score)

        # Display the results for each test image
        st.subheader(f"Results for: {test_file.name}")
        st.image(ref_img, caption="Original PAN Card", use_container_width=True, channels="GRAY")
        st.image(diff, caption="SSIM Difference Image", use_container_width=True, channels="GRAY")
        st.image(thresh_img, caption="Thresholded Tampering Detection Image", use_container_width=True, channels="GRAY")
        st.image(result_img, caption="Detected Tampered Sections", use_container_width=True)

        st.write(f"**SSIM Score:** {score:.4f}")
        st.write(f"**Result:** {result}")
        st.write(f"**Reason:** {reason}")

        if tampered_areas:
            st.write("**Tampered Sections Detected:**")
            for i, (x, y, w, h) in enumerate(tampered_areas, 1):
                st.write(f"❌ Section {i}: Tampered (Bounding Box: ({x}, {y}, {w}, {h}))")
        else:
            st.write("✅ No tampering detected.")

        st.markdown("---")
