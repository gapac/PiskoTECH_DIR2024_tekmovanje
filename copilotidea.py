import cv2
import numpy as np


# Load an image from file
image = cv2.imread('slika.jpg')
cv2.imshow('Image', image)
cv2.waitKey(0)


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
gray = cv2.GaussianBlur(gray, (5, 5), 0)
#show image
cv2.imshow('Image', gray)
cv2.waitKey(0)

#treshold the image
_, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)


# Apply edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours in the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over the contours
for contour in contours:
    # Get the rectangle that encloses the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Check if the contour is a square
    if 0.9 < w/h < 1.1:  # Adjust these values based on your requirement
        # Draw the rectangle on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()