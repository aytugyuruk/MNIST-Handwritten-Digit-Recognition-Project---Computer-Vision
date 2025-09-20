import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from skimage.feature import hog

#========================================================
train_file_path = '/Users/aytug/Desktop/handwritten_digit_recognition/data/train-images.idx3-ubyte'
train_labels_file_path = '/Users/aytug/Desktop/handwritten_digit_recognition/data/train-labels.idx1-ubyte'
test_file_path = '/Users/aytug/Desktop/handwritten_digit_recognition/data/t10k-images.idx3-ubyte'
test_labels_file_path = '/Users/aytug/Desktop/handwritten_digit_recognition/data/t10k-labels.idx1-ubyte'
#========================================================
# --- Train set ---
with open(train_file_path, 'rb') as f:
    f.read(16)
    train_data = f.read()

x_train = np.frombuffer(train_data, dtype=np.uint8)
x_train = x_train.reshape(-1, 28, 28) / 255.0

with open(train_labels_file_path, 'rb') as f:
    f.read(8)
    y_train = np.frombuffer(f.read(), dtype=np.uint8)

# --- Test set ---
with open(test_file_path, 'rb') as f:
    f.read(16)
    test_data = f.read()

x_test = np.frombuffer(test_data, dtype=np.uint8)
x_test = x_test.reshape(-1, 28, 28) / 255.0

with open(test_labels_file_path, 'rb') as f:
    f.read(8)
    y_test = np.frombuffer(f.read(), dtype=np.uint8)

# --- Flatten images ---
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat  = x_test.reshape(x_test.shape[0], -1)

# --- HOG feature extraction ---
def extract_hog_features(images):
    features = []
    for img in images:
        feature = hog(img, orientations=9, pixels_per_cell=(8,8),
                      cells_per_block=(2,2), block_norm='L2-Hys')
        features.append(feature)
    return np.array(features)

x_train_hog = extract_hog_features(x_train)
x_test_hog  = extract_hog_features(x_test)

# --- Ham piksel + HOG birleştirme ---
x_train_combined = np.hstack([x_train_flat, x_train_hog])
x_test_combined  = np.hstack([x_test_flat, x_test_hog])

# --- SVM modeli ---
svm_model = SVC(verbose=True)
svm_model.fit(x_train_combined, y_train)

# --- Tahmin ve sonuç ---
y_pred_svm = svm_model.predict(x_test_combined)
print("SVM Test Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# --- Modeli kaydet ---
dump(svm_model, 'svm_mnist_pixel_hog_model.joblib')
print("Model kaydedildi: svm_mnist_pixel_hog_model.joblib")