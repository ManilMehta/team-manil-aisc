import cv2
import numpy as np

# Load the image
image = cv2.imread(
    "C:/Users/micah/OneDrive/Documents/Spring AISC Project/team-manil-aisc/data/melanoma_cancer_dataset/train/benign/melanoma_0.jpg"
)
if image is None:
    raise FileNotFoundError("Could not read sample image.")

# Resize to 224x224
resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

# Pad shorter image so heights match for side-by-side display
target_h = max(image.shape[0], resized_image.shape[0])


def pad_to_height(img, h):
    pad = h - img.shape[0]
    if pad <= 0:
        return img
    return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


original_padded = pad_to_height(image, target_h)
resized_padded = pad_to_height(resized_image, target_h)
comparison = np.hstack([original_padded, resized_padded])

cv2.imshow("Original (left) vs Resized 224x224 (right)", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()

