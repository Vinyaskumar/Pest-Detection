import cv2
import numpy as np

def pest_detection(image_path, background_path):
    # Load the images
    image = cv2.imread(image_path)
    background = cv2.imread(background_path)

    # Convert the images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Perform image subtraction
    subtracted_image = cv2.absdiff(gray_image, gray_background)
    cv2.imshow('img',subtracted_image)
    # Apply thresholding to obtain binary image
    threshold = cv2.threshold(subtracted_image, 30, 255, cv2.THRESH_BINARY)

    # Apply morphological operations for noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours of detected objects
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around pests
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Minimum area threshold to filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image

# Specify the paths to your image and background image
image_path = "bug1.jpg"
background_path = "leaf1.jpg"

# Perform pest detection
result = pest_detection(image_path, background_path)

# Display the result
cv2.imshow("Pest Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
