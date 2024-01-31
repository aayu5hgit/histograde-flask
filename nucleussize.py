# nucleussize_normalized.py
from flask import Flask, jsonify, request
import numpy as np
import cv2
import base64
from PIL import Image
from normalize_image import normalize_image  # Import the normalization function

app = Flask(__name__)

def apply_color_mask(image, low_range, high_range):
    lower_bound = np.array(low_range, dtype=np.uint8)
    upper_bound = np.array(high_range, dtype=np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def find_draw_nuclei_boundaries_and_get_sizes(image, min_area=50):
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    _, binary_mask = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    nuclei_count = 0
    nuclei_sizes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 1)
            nuclei_count += 1
            nuclei_sizes.append(area)

    nuclei_sizes_array = np.array(nuclei_sizes)

    return result_image, nuclei_count, nuclei_sizes_array, contours

def calculate_average_nucleus_size(image_height, nuclei_contours, nuclei_sizes):
    section_height = image_height // 3
    top_section_sizes = []
    middle_section_sizes = []
    bottom_section_sizes = []

    for contour, size in zip(nuclei_contours, nuclei_sizes):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        if 0 <= cy < section_height:
            top_section_sizes.append(size)
        elif section_height <= cy < 2 * section_height:
            middle_section_sizes.append(size)
        elif 2 * section_height <= cy < image_height:
            bottom_section_sizes.append(size)

    average_top_section_size = np.mean(top_section_sizes) if top_section_sizes else 0
    average_middle_section_size = np.mean(middle_section_sizes) if middle_section_sizes else 0
    average_bottom_section_size = np.mean(bottom_section_sizes) if bottom_section_sizes else 0

    return average_top_section_size, average_middle_section_size, average_bottom_section_size

def draw_horizontal_lines(image, section_height):
    line_color = (0, 255, 0)
    line_thickness = 2

    cv2.line(image, (0, section_height), (image.shape[1], section_height), line_color, line_thickness)
    cv2.line(image, (0, 2 * section_height), (image.shape[1], 2 * section_height), line_color, line_thickness)

def process_nucleus_image(normalized_image_bytes):
    original_image = cv2.imdecode(np.frombuffer(normalized_image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Specify the path to the reference image for normalization
    ref_image_file = './dataset/Severe/IMG_20230117_125305.jpg'

    # Normalize colors of the input image using the normalization utility
    normalized_image = normalize_image(original_image, ref_image_file)

    # Define color ranges for nucleus detection (adjust these ranges based on your specific case)
    nucleus_low_range = (20, 0, 45)
    nucleus_high_range = (120, 45, 90)

    # Apply color range masking for nucleus detection
    nucleus_masked_image = apply_color_mask(normalized_image, nucleus_low_range, nucleus_high_range)

    # Convert the masked image to grayscale
    gray_nucleus_masked_image = cv2.cvtColor(nucleus_masked_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create a binary mask of the detected nuclei
    _, nucleus_binary_mask = cv2.threshold(gray_nucleus_masked_image, 0, 255, cv2.THRESH_BINARY)

    # Find contours of the detected nuclei
    nucleus_contours, _ = cv2.findContours(nucleus_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original normalized image to overlay nucleus contours
    image_with_nucleus_contours = normalized_image.copy()

    # Draw contours around the detected nuclei in red
    cv2.drawContours(image_with_nucleus_contours, nucleus_contours, -1, (0, 0, 255), 1)

    draw_horizontal_lines(image_with_nucleus_contours, image_with_nucleus_contours.shape[0] // 3)

    # Encode the image with nucleus contours to base64 for JSON response
    _, img_encoded_nucleus_contours = cv2.imencode('.png', image_with_nucleus_contours)
    img_base64_nucleus_contours = base64.b64encode(img_encoded_nucleus_contours).decode('utf-8')

    # Encode the original image to base64 for JSON response
    _, img_encoded_original = cv2.imencode('.png', original_image)
    img_base64_original = base64.b64encode(img_encoded_original).decode('utf-8')

    response_data = {
        'nucleusContoursImage': img_base64_nucleus_contours,
        'nucleusContoursCount': len(nucleus_contours),
        'originalImage': img_base64_original
    }
    return response_data



if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
