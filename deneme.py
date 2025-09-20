from joblib import load
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog

# --- Modeli yükle ---
model = load('/Users/aytug/Desktop/handwritten_digit_recognition/svm_mnist_pixel_hog_model.joblib')

# --- Test resmi oku ---
img = cv2.imread('/Users/aytug/Desktop/handwritten_digit_recognition/numbers/7-Seven.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Threshold (binary) ve invert et ---
_, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

# --- 28x28 boyutuna getir ---
resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

# --- Normalize (0-1 aralığına) ---
img_array = np.array(resized) / 255.0

# --- Ham pixel flatten ---
img_flat = img_array.reshape(1, -1)

# --- HOG feature çıkar ---
img_hog = hog(img_array, orientations=9, pixels_per_cell=(8,8),
              cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
img_hog = img_hog.reshape(1, -1)

# --- Ham pixel + HOG birleştir ---
img_combined = np.hstack([img_flat, img_hog])

# --- Tahmin al ---
predicted_digit = model.predict(img_combined)
print("Tahmin Edilen Rakam:", predicted_digit[0])

# --- Görüntüyü göster ---
plt.imshow(resized, cmap='gray')
plt.title(f"Tahmin: {predicted_digit[0]}")
plt.show()