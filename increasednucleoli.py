from flask import Flask, request, jsonify
import cv2
import numpy as np
import urllib.request


# Function to apply color range masking
def apply_color_mask(image, low_range, high_range):
    lower_bound = np.array(low_range, dtype=np.uint8)
    upper_bound = np.array(high_range, dtype=np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# Function to apply connected component labeling
def label_connected_components(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    return num_labels, labels, stats, centroids

# Function to detect and count black spots (objects) in an image
def count_black_spots(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary mask of the black spots
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the black spots
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the detected black spots
    spot_count = len(contours)

    # Return the count
    return spot_count

def analyze_nucleoli(image_bytes):
    try:
        # Read the image data
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Check if the image is grayscale
        if len(image.shape) == 2:
            # Convert grayscale image to 3-channel (RGB)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Adjusted lower and upper bounds for cell detection
        cell_low_range = (52, 52, 52)
        cell_high_range = (255, 255, 255)

        # Apply the color range masking with the adjusted range for cell detection
        masked_image = apply_color_mask(image, cell_low_range, cell_high_range)

        # Convert the masked image to grayscale
        gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Count the number of cells with more than one nucleus
        nucleus_count = count_black_spots(gray_masked_image)

        # Return the result as a dictionary
        return {'increased_nucleoli': nucleus_count}

    except Exception as e:
        # Return the error message
        return {'error': str(e)}


