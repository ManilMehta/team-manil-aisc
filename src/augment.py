import cv2
import numpy as np
from pathlib import Path

try:
    import albumentations as A
    _HAS_ALBUMENTATIONS = True
except ImportError:
    A = None
    _HAS_ALBUMENTATIONS = False

TARGET_SIZE = (224, 224)
MODEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MODEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_image(image):
    """Resize image to 224x224."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def load_image_opencv(image_path, rgb=True):
    """Load an image with OpenCV."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def normalize_image(image):
    """Normalize RGB image using ImageNet mean/std."""
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return (image - MODEL_MEAN) / MODEL_STD


def preprocess_image(image, normalize=True):
    """Resize to 224x224 and optionally normalize."""
    resized = resize_image(image)
    if normalize:
        return normalize_image(resized)
    return resized


def show_original_vs_resized(image_path):
    """Display original and resized images side by side using OpenCV."""
    original_bgr = load_image_opencv(image_path, rgb=False)
    resized_bgr = cv2.resize(original_bgr, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    height, width = original_bgr.shape[:2]
    display_height = TARGET_SIZE[1]
    display_width = max(1, int(width * (display_height / height)))
    original_display = cv2.resize(original_bgr, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    cv2.putText(original_display, "Original", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(resized_bgr, "Resized 224x224", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    comparison = np.hstack((original_display, resized_bgr))
    cv2.imshow("Original vs Resized 224x224", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def augment_image_uint8(image, flip_prob=0.5, brightness_limit=0.2):
    """Resize, flip, and adjust brightness. Returns uint8 image."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    image = resize_image(image)
    
    if _HAS_ALBUMENTATIONS:
        transform = A.Compose([
            A.HorizontalFlip(p=flip_prob),
            A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=0.0, p=1.0),
        ])
        image = transform(image=image)["image"]
    else:
        if np.random.rand() < flip_prob:
            image = cv2.flip(image, 1)
        brightness_factor = 1.0 + np.random.uniform(-brightness_limit, brightness_limit)
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    return np.clip(image, 0, 255).astype(np.uint8)


def augment_image(image, flip_prob=0.5, brightness_limit=0.2, normalize=True):
    """Apply Albumentations resize, flip, brightness, then optionally normalize."""
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("Albumentations is required for augment_image(). Install it with: pip install albumentations")

    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    transform = A.Compose([
        A.Resize(TARGET_SIZE[1], TARGET_SIZE[0], interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=flip_prob),
        A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=0.0, p=1.0),
    ])
    augmented = transform(image=image)["image"]

    if normalize:
        return normalize_image(augmented)
    return augmented


def load_dataset(root_dir, verbose=True):
    """Load images from benign/malignant subdirectories."""
    classes = [("benign", 0), ("malignant", 1)]
    X, y = [], []
    root_dir = Path(root_dir)
    
    all_paths = []
    for class_name, label in classes:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Folder not found: {class_dir}")
        for pattern in ["*.jpg", "*.jpeg", "*.png"]:
            all_paths.extend((p, label) for p in sorted(class_dir.glob(pattern)))
    
    if verbose:
        print(f"Found {len(all_paths)} images in {root_dir}")
    
    for idx, (image_path, label) in enumerate(all_paths, start=1):
        img = cv2.imread(str(image_path))
        if img is None:
            if verbose and idx % 500 == 0:
                print(f"  loaded {idx}/{len(all_paths)}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = resize_image(img_rgb)
        X.append(img_resized)
        y.append(label)
        
        if verbose and idx % 500 == 0:
            print(f"  loaded {idx}/{len(all_paths)}")
    
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)


def save_augmented_images(X, y, output_root, prefix="aug"):
    """Save augmented images with aug_ prefix."""
    output_root = Path(output_root)
    for cls_name in ["benign", "malignant"]:
        (output_root / cls_name).mkdir(parents=True, exist_ok=True)
    
    for idx, (image, label) in enumerate(zip(X, y), start=1):
        cls_name = ["benign", "malignant"][int(label)]
        out_path = output_root / cls_name / f"{prefix}_{idx:06d}.jpg"
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
    
    return output_root


def augment_dataset(X, y, flip_prob=0.5, brightness_limit=0.2):
    """Create augmented copies (original + augmented)."""
    X = np.array([resize_image(img) if img.shape[:2] != TARGET_SIZE else img for img in X], dtype=np.uint8)
    X_aug = [augment_image_uint8(x, flip_prob, brightness_limit) for x in X]
    return np.vstack((X, X_aug)), np.append(y, y)


def augment_dataset_inplace(input_root, output_root, flip_prob=0.5, brightness_limit=0.2, verbose=True):
    """Augment and save images with original filenames."""
    input_root, output_root = Path(input_root), Path(output_root)
    for cls_name in ["benign", "malignant"]:
        (output_root / cls_name).mkdir(parents=True, exist_ok=True)
    
    for cls_id, cls_name in enumerate(["benign", "malignant"]):
        files = sorted((input_root / cls_name).glob("*.jpg")) + sorted((input_root / cls_name).glob("*.png"))
        if verbose:
            print(f"Processing {cls_name}: {len(files)} images")
        
        for idx, img_path in enumerate(files, 1):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_aug = augment_image_uint8(img_rgb, flip_prob, brightness_limit)
                    cv2.imwrite(str(output_root / cls_name / img_path.name), cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))
                    if verbose and ((idx % 500 == 0) or (idx == len(files))):
                        print(f"  processed {idx}/{len(files)}")
            except Exception as e:
                if verbose:
                    print(f"  Error on {img_path.name}: {e}")





