import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Function to detect if the given card is a PAN card
def is_pan_card(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use OCR to extract text
    text = pytesseract.image_to_string(gray)

    # Check for specific keywords in the text
    keywords = ['Income Tax Department', 'Permanent Account Number', 'INCOME TAX DEPARTMENT', 'GOVT. OF INDIA']
    for keyword in keywords:
        if keyword.lower() in text.lower():
            return True
    return False

# Function to detect tampering and calculate the percentage of tampered area
def detect_tampering(original_img, test_img):
    # Resize images to the same dimensions
    original_img_resized = cv2.resize(original_img, (test_img.shape[1], test_img.shape[0]))

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the images
    diff_img = cv2.absdiff(original_gray, test_gray)

    # Threshold the difference image to get the regions with significant changes
    _, thresh_img = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of tampered area
    tampered_area = np.sum(thresh_img > 0)
    total_area = thresh_img.size
    tampered_percentage = (tampered_area / total_area) * 100

    return tampered_percentage, thresh_img

# Streamlit App
st.title("PAN Card Tampering Detection")

# Load the original image (stored in your GitHub repository or locally)
original_img_path = 'https://github.com/YuvaAvuy/pan_card_original/blob/main/pan%201.jpg'
original_img = cv2.imread(original_img_path)

# Upload the test image
uploaded_file = st.file_uploader("Upload the test PAN card image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    test_img = Image.open(uploaded_file)
    test_img = np.array(test_img)

    # Check if the test image is a PAN card
    if not is_pan_card(test_img):
        st.error("The provided image is not a PAN card.")
    else:
        # Detect tampering and calculate percentage
        tampered_percentage, tampered_img = detect_tampering(original_img, test_img)

        # Draw contours on the tampered image for visualization
        contours, _ = cv2.findContours(tampered_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        test_img_contours = test_img.copy()
        cv2.drawContours(test_img_contours, contours, -1, (0, 255, 0), 2)

        # Display the images
        st.image([cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(test_img_contours, cv2.COLOR_BGR2RGB)],
                 caption=["Original PAN Card", "Test PAN Card", "Detected Tampering"],
                 use_column_width=True)

        # Display the tampering analysis result
        if tampered_percentage > 0:
            st.warning(f"The provided PAN card is tampered with {tampered_percentage:.2f}% of its area.")
        else:
            st.success("The provided PAN card is not tampered.")

