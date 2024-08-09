import cv2
import numpy as np
import pytesseract
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import requests

# Function to load the original image from GitHub
def load_original_image_from_github(url):
    response = requests.get(url)
    img = Image.open(response.content)
    return np.array(img)

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

# Function to analyze the card
def analyze_card(original_img, test_img):
    # Check if the test image is a PAN card
    if not is_pan_card(test_img):
        return "The provided image is not a PAN card."

    # Detect tampering and calculate percentage
    tampered_percentage, tampered_img = detect_tampering(original_img, test_img)

    # Draw contours on the tampered image for visualization
    contours, _ = cv2.findContours(tampered_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    test_img_contours = test_img.copy()
    cv2.drawContours(test_img_contours, contours, -1, (0, 255, 0), 2)

    # Display the images and tampering result
    st.image([cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 
              cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), 
              cv2.cvtColor(test_img_contours, cv2.COLOR_BGR2RGB)], 
              caption=["Original PAN Card", "Test PAN Card", "Detected Tampering"], 
              width=300)

    # Return the tampering analysis result
    if tampered_percentage > 0:
        return f"The provided PAN card is tampered with {tampered_percentage:.2f}% of its area."
    else:
        return "The provided PAN card is not tampered."

# Streamlit UI
st.title("PAN Card Tampering Detection")

# URL of the original image stored in GitHub
github_url = "https://github.com/YuvaAvuy/pan_card_original/blob/main/1.jpg"

# Load the original PAN card image
original_img = load_original_image_from_github(github_url)

# Upload test image
uploaded_test_img = st.file_uploader("Upload Test PAN Card Image", type=["jpg", "jpeg", "png"])

if uploaded_test_img is not None:
    # Convert uploaded image to OpenCV format
    test_img = np.array(Image.open(uploaded_test_img))

    # Analyze the card
    result = analyze_card(original_img, test_img)

    # Display the result
    st.write(result)
