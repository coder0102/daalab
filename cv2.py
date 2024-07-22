Lab 7 -- Quadrant
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("rabbit.jpg")
height,width=img.shape[:2]
quad1 = img[:height//2,:width//2]
quad2 = img[:height//2, width//2:]
quad3 = img[height//2:, :width//2]
quad4 = img[height//2:, width//2:]
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(quad1)
plt.title("1")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(quad2)
plt.title("2")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(quad3)
plt.title("3")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(quad4)
plt.title("4")
plt.axis("off")
plt.show()

Lab 8 -- Translation
import cv2
import numpy as np

# Load an image
image = cv2.imread('rabbit.jpg')

# User inputs
tx = int(input("Enter translation in x direction: "))
ty = int(input("Enter translation in y direction: "))
fx = float(input("Enter scaling factor for x: "))
fy = float(input("Enter scaling factor for y: "))
angle = float(input("Enter rotation angle in degrees: "))

# Translation
def translate(image, x, y):
    rows,col = image.shape[:2]
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, matrix, (rows,col))

# Scaling
def scale(image, fx, fy):
    return cv2.resize(image, None, fx=fx, fy=fy)

# Rotation
def rotate(image, angle):
    height,width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))

# Apply transformations
translated_image = translate(image, tx, ty)
scaled_image = scale(image, fx, fy)
rotated_image = rotate(image, angle)

# Display results
cv2.imshow('Translated Image', translated_image)
cv2.imshow('Scaled Image', scaled_image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

LAB 9 - EGDES
import cv2
import numpy as np
img = cv2.imread("Face.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100,200)
kernel = np.ones((5,5),np.float32)/25
textures = cv2.filter2D(gray,-1,kernel)
cv2.imshow("Gray",gray)
cv2.imshow("Edges",edges)
cv2.imshow("Textures",textures)
cv2.waitKey(0)
cv2.destroyAllWindows()


Lab 10 -- Blur
import cv2

# Load an image
image = cv2.imread('Face.jpeg')

# Apply smoothing using a Gaussian filter
def smooth_image(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Apply blurring using a median filter
def blur_image(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

# User inputs
kernel_size_smooth = int(input("Enter kernel size for smoothing (must be odd): "))
kernel_size_blur = int(input("Enter kernel size for blurring (must be odd): "))

# Apply transformations
smoothed_image = smooth_image(image, kernel_size_smooth)
blurred_image = blur_image(image, kernel_size_blur)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Smoothed Image', smoothed_image)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Lab 11 -- Contour
import cv2
import numpy as np

image_path = "rabbit.jpg"
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary threshold
_,binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours,_= cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Display the image with contours
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

Lab 12 -- Face Detect
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
image_path="Face.jpeg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
for(x,y,w,h) in faces:
  cv2.rectangle(img, (x, y),(x + w, y + h),(255,0,0),2)
cv2.imwrite('detected_face.jpeg',img)
cv2.imshow('Detcted Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


