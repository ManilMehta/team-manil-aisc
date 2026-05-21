import cv2
import numpy as np
from pathlib import Path

try:
    import albumentations as A
    _HAS_ALBUMENTATIONS = True
except ImportError:
    A = None
    _HAS_ALBUMENTATIONS = False

SAMPLE_PATH = (
    "C:/Users/micah/OneDrive/Documents/Spring AISC Project/team-manil-aisc/"
    "data/melanoma_cancer_dataset/train/benign/melanoma_0.jpg"
)
TARGET_SIZE = (224, 224)
MODEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MODEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if _HAS_ALBUMENTATIONS:
    resize_transform = A.Compose([
        A.Resize(TARGET_SIZE[0], TARGET_SIZE[1], interpolation=cv2.INTER_LINEAR),
    ])
    normalize_transform = A.Compose([
        A.Normalize(mean=MODEL_MEAN, std=MODEL_STD, max_pixel_value=255.0),
    ])
else:
    resize_transform = None
    normalize_transform = None


def load_rgb_image(path):
    """Load image and convert from BGR to RGB."""
    # Prefer cv2.imread first (fast), but fall back to Windows-friendly
    # read methods if it fails (handles unicode/long paths or corrupted files).
    # 1) cv2.imread
    image = cv2.imread(path)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2) np.fromfile + cv2.imdecode (works better on Windows with long/unicode paths)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        # fall through to PIL attempt
        pass

    # 3) PIL fallback (if available)
    try:
        from PIL import Image

        pil = Image.open(path).convert("RGB")
        return np.asarray(pil)
    except Exception:
        pass

    # If all methods failed, raise a descriptive error so caller can decide to skip
    raise FileNotFoundError(f"Could not read image from {path} using cv2, imdecode, or PIL")


def preprocess_image(image):
    """Resize image to TARGET_SIZE using Albumentations when available."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if _HAS_ALBUMENTATIONS:
        return resize_transform(image=image)["image"]
    return cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def normalize_image(image):
    """Apply ImageNet normalization using Albumentations when available."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if _HAS_ALBUMENTATIONS:
        return normalize_transform(image=image)["image"]
    image = image.astype(np.float32) / 255.0
    return (image - MODEL_MEAN) / MODEL_STD


def denormalize_image(image):
    """Reverse normalization to scale back to [0, 255] uint8."""
    image = image.astype(np.float32) * MODEL_STD + MODEL_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def resize_image(image):
    """Resize only, return as uint8."""
    return preprocess_image(image).astype(np.uint8)


def augment_image_uint8(image, flip_prob=0.5, brightness_limit=0.2):
    """Apply augmentation and return a uint8 image for dataset augmentation."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    image = preprocess_image(image)

    if _HAS_ALBUMENTATIONS:
        transform = A.Compose([
            A.HorizontalFlip(p=flip_prob),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=0.0,
                p=1.0,
            ),
        ])
        image = transform(image=image)["image"]
    else:
        if np.random.rand() < flip_prob:
            image = cv2.flip(image, 1)

        brightness_factor = 1.0 + np.random.uniform(-brightness_limit, brightness_limit)
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    return np.clip(image, 0, 255).astype(np.uint8)


def augment_image(image, flip_prob=0.5, brightness_limit=0.2):
    """
    Apply augmentation: resize, random flip, brightness adjustment, and normalize.

    Args:
        image: Input RGB image (uint8)
        flip_prob: Probability of horizontal flip (0.0 to 1.0)
        brightness_limit: Brightness adjustment range (0.0 to 1.0)

    Returns:
        Augmented and normalized image
    """
    image = augment_image_uint8(image, flip_prob, brightness_limit)
    return normalize_image(image)


def augment_dataset(X, y, flip_prob=0.5, brightness_limit=0.2):
    """Create augmented copies of X and y using flip/brightness."""
    # Ensure the base dataset is resized before stacking.
    X = np.array([resize_image(img) if img.shape[:2] != TARGET_SIZE else img for img in X], dtype=np.uint8)

    print(f"Augmenting {len(X)} images...")
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        X_augmented.append(augment_image_uint8(X[i], flip_prob, brightness_limit))
        y_augmented.append(y[i])
        if (i + 1) % 500 == 0 or i == len(X) - 1:
            print(f"  augmented {i + 1}/{len(X)} images")

    X_augmented = np.array(X_augmented, dtype=np.uint8)
    y_augmented = np.array(y_augmented, dtype=y.dtype)

    X_combined = np.vstack((X, X_augmented))
    y_combined = np.append(y, y_augmented)
    return X_combined, y_combined


def augment_dataset_pair(X, X_g, y, flip_prob=0.5, brightness_limit=0.2):
    """Augment a pair of parallel image arrays (X and X_g) with shared labels."""
    X = np.array([resize_image(img) if img.shape[:2] != TARGET_SIZE else img for img in X], dtype=np.uint8)
    X_g = np.array([resize_image(img) if img.shape[:2] != TARGET_SIZE else img for img in X_g], dtype=np.uint8)

    X_augmented = []
    X_g_augmented = []
    y_augmented = []

    for i in range(len(X)):
        X_augmented.append(augment_image_uint8(X[i], flip_prob, brightness_limit))
        X_g_augmented.append(augment_image_uint8(X_g[i], flip_prob, brightness_limit))
        y_augmented.append(y[i])

    X_augmented = np.array(X_augmented, dtype=np.uint8)
    X_g_augmented = np.array(X_g_augmented, dtype=np.uint8)
    y_augmented = np.array(y_augmented, dtype=y.dtype)

    X_combined = np.vstack((X, X_augmented))
    X_g_combined = np.vstack((X_g, X_g_augmented))
    y_combined = np.append(y, y_augmented)
    return X_combined, X_g_combined, y_combined


def load_dataset(root_dir, verbose=True):
    """Load all images and labels from a train folder structure."""
    classes = [("benign", 0), ("malignant", 1)]
    X = []
    y = []
    root_dir = Path(root_dir)

    all_paths = []
    for class_name, label in classes:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Training folder not found: {class_dir}")

        for pattern in ["*.jpg", "*.jpeg", "*.png"]:
            for image_path in sorted(class_dir.glob(pattern)):
                all_paths.append((image_path, label))

    if verbose:
        print(f"Found {len(all_paths)} training images in {root_dir}")

    for idx, (image_path, label) in enumerate(all_paths, start=1):
        try:
            image = load_rgb_image(str(image_path))
        except Exception as e:
            print(f"Warning: skipping unreadable image {image_path}: {e}")
            continue

        try:
            image = resize_image(image)
        except Exception as e:
            print(f"Warning: failed to resize image {image_path}: {e}")
            continue

        X.append(image)
        y.append(label)

        if verbose and idx % 500 == 0:
            print(f"  loaded {idx}/{len(all_paths)} images")

    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)


def save_augmented_images(X, y, output_root, prefix="aug"):
    """Save augmented images to disk in class subfolders."""
    output_root = Path(output_root)
    classes = {0: "benign", 1: "malignant"}
    for cls_name in classes.values():
        (output_root / cls_name).mkdir(parents=True, exist_ok=True)

    for idx, (image, label) in enumerate(zip(X, y), start=1):
        cls_name = classes.get(int(label), str(label))
        out_path = output_root / cls_name / f"{prefix}_{idx:06d}.jpg"
        bgr = rgb_to_bgr(image)
        if not cv2.imwrite(str(out_path), bgr):
            raise IOError(f"Failed to save augmented image {out_path}")

    return output_root


def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def stack_images(images, labels):
    """Stack images horizontally with labels. All images must be same height."""
    rows = []
    for image, label in zip(images, labels):
        # Ensure image is uint8 and correct size
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        # Resize to TARGET_SIZE if not already
        if image.shape[0] != TARGET_SIZE[0] or image.shape[1] != TARGET_SIZE[1]:
            image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        
        bgr = rgb_to_bgr(image)
        label_img = np.full((30, bgr.shape[1], 3), 0, dtype=np.uint8)
        cv2.putText(
            label_img,
            label,
            (5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([label_img, bgr]))
    return np.hstack(rows)


if __name__ == "__main__":
    dataset_root = Path(__file__).resolve().parents[1] / "data" / "melanoma_cancer_dataset" / "train"
    X, y = load_dataset(dataset_root, verbose=True)

    print(f"Loaded training images: {len(X)}")
    print(f"Loaded labels: {y.shape}")
    print("Starting augmentation of the full training dataset...")

    X_combined, y_combined = augment_dataset(X, y)
    print(f"Augmented dataset: {X_combined.shape} (original + augmented)")
    print(f"Augmented labels: {y_combined.shape}")
    print(f"Original count: {len(X)}")
    print(f"Augmented count: {len(X_combined) - len(X)}")
    print(f"Total examples after augmentation: {len(X_combined)}")

    augmented_root = Path(__file__).resolve().parents[1] / "data" / "melanoma_cancer_dataset" / "train_augmented"
    augmented_images = X_combined[len(X):]
    augmented_labels = y_combined[len(X):]
    print(f"Saving {len(augmented_images)} augmented images to {augmented_root} ...")
    save_augmented_images(augmented_images, augmented_labels, augmented_root)
    print(f"Saved augmented images to {augmented_root}")

