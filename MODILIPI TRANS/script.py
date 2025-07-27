import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in color
image_path = "modi_lipi_sample.jpg"  
image = cv2.imread(image_path)

# Convert to LAB color space (for better contrast enhancement)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_enhanced = clahe.apply(l_channel)

# Merge the enhanced L-channel back with A and B channels
lab_enhanced = cv2.merge((l_enhanced, a, b))
color_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

# Apply sharpening to enhance the text edges
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(color_enhanced, -1, kernel)

# Apply mild denoising to avoid blur
denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)

# Show results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2RGB))
plt.title("Contrast Enhanced")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
plt.title("Final Restored Image")

plt.show()

# Save the processed image
cv2.imwrite("modi_lipi_restored.jpg", denoised)




