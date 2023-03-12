import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from skimage.feature import hog
import os
# print("Loading dataset...")
# mnist = fetch_openml('mnist_784', version=1)
# X = mnist.data
# y = mnist.target.astype(int)



dataset_dir = "dataset"

# Get a list of all image files in the dataset directory
image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".jpg")]

# Create an empty list to store the images
images = []

# Load each image using OpenCV and append to the list
for file in image_files:
    image = cv2.imread(file,0)
    images.append(image)

# Convert the list to a numpy array
# images = np.array(images)
print(images)
print("Extracting features...")
features = []
for image in images:
    print(image.shape)
    fd = hog(image.reshape((1080,1920)), orientations=8, pixels_per_cell=(4, 4),
             cells_per_block=(1, 1), visualize=False)
    features.append(fd)

X_train, X_test, y_train, y_test = train_test_split(features, [29,], test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("Starting video stream...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
    threshold = cv2.erode(threshold, np.ones((2, 2), np.uint8), iterations=1)
    threshold = cv2.dilate(threshold, np.ones((2, 2), np.uint8), iterations=1)
    fd = hog(threshold, orientations=8, pixels_per_cell=(4, 4),
             cells_per_block=(1, 1), visualize=False)
    fd = np.array(fd).reshape(1, -1)
    label = knn.predict(fd)[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(label), (10, 50), font, 2, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()