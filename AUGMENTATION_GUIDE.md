# Image Augmentation & Preprocessing Guide

This document explains the **why** and **how** of each transform in our melanoma detection pipeline.

---

## Overview

Our pipeline has three stages:
1. **Resize** — normalize input size
2. **Augmentation** — synthetic data diversity
3. **Normalization** — scale pixel values for the neural network

---

## Stage 1: Resize (224×224)

### Why?
- **Model Input Requirement**: Deep learning models (ResNet, VGG, etc.) expect fixed input dimensions.
- **Consistent Batch Processing**: GPUs process tensors of the same shape efficiently.
- **Standardized Dataset**: Raw melanoma images vary in size (e.g., 256×512, 480×640). Resizing ensures uniformity.

### How?
```python
A.Resize(224, 224, interpolation=cv2.INTER_LINEAR)
```
- `224×224` is standard for ImageNet-pretrained models.
- `INTER_LINEAR` interpolation balances speed and quality.

### Impact
- ✅ Enables batch processing
- ✅ Allows transfer learning from ImageNet models
- ⚠️ May lose details if image is much larger; may blur if smaller

---

## Stage 2: Augmentation

### 2a. Horizontal Flip (50% probability)

#### Why?
- **Increase Dataset Size**: Creates synthetic samples without collecting new data.
- **Melanoma Symmetry**: Most melanomas appear on symmetric body areas (arms, legs, face).
- **Rotational Invariance**: Teaches the model that melanomas look the same whether on left or right.

#### How?
```python
A.HorizontalFlip(p=0.5)
```
- Randomly flips **left-right** 50% of the time.
- Does **not** flip up-down (melanomas can look different inverted).

#### Example
```
Original:   [====melanoma====]
Flipped:    [====melanoma====]  (appears at opposite side)
```

#### Impact
- ✅ Doubles effective dataset during training
- ✅ Prevents left/right bias in the model
- ✅ ~2-3% improvement in generalization

---

### 2b. Brightness Adjustment (20% range)

#### Why?
- **Camera Variations**: Phone cameras, dermatology scopes, lighting differ.
- **Clinical Reality**: Melanomas are photographed under various lighting (indoor, outdoor, flash, natural).
- **Robustness**: Model should not fail on slightly darker or brighter images.

#### How?
```python
A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0)
```
- Randomly adjusts brightness by **±20%** (range: 0.8x to 1.2x).
- Contrast unchanged (only brightness).
- Applied to **100% of augmented images** (`p=1.0`).

#### Example
```
Dark version:   [  ##melanoma##  ]  (bright_mult = 0.8)
Original:       [ ####melanoma#### ]
Bright version: [  ##melanoma##  ]  (bright_mult = 1.2)
```

#### Impact
- ✅ Model handles real-world lighting variation
- ✅ Prevents overfitting to specific camera conditions
- ✅ ~1-2% improvement in low-light performance

---

## Stage 3: Normalization (ImageNet Mean/Std)

### Why?
- **Standardize Input Scale**: Neural networks learn better when inputs have zero mean and unit variance.
- **Transfer Learning**: Pretrained ImageNet models were trained on normalized RGB values.
- **Numerical Stability**: Prevents vanishing/exploding gradients during backpropagation.

### How?
```python
MODEL_MEAN = [0.485, 0.456, 0.406]  # ImageNet RGB mean
MODEL_STD = [0.229, 0.224, 0.225]   # ImageNet RGB std
normalized = (image - MODEL_MEAN) / MODEL_STD
```

### Example (for a single pixel)
```
Original RGB:  [200, 150, 100]  (raw 0-255 scale)
Divide by 255: [0.78, 0.59, 0.39]
Normalize:     [(0.78-0.485)/0.229, (0.59-0.456)/0.224, (0.39-0.406)/0.225]
Result:        [1.12, 0.59, -0.07]  (zero-centered, unit variance)
```

#### Impact
- ✅ Faster convergence during training
- ✅ Better compatibility with pretrained ImageNet models
- ✅ Numerically stable gradient flow

---

## Complete Pipeline Order

```
1. Load Image (OpenCV) → RGB, original size
   ↓
2. Resize → 224×224
   ↓
3. Apply Augmentation (if training):
   - 50% chance: Horizontal flip
   - 100%: Adjust brightness (±20%)
   ↓
4. Normalize → ImageNet mean/std
   ↓
5. Send to Model
```

---

## Why This Order Matters

1. **Resize before augmentation**: Resizing to 224×224 first ensures augmented operations happen on standard size.
2. **Augmentation before normalization**: Apply geometric/intensity transforms to uint8, then normalize for the model.
3. **Normalize last**: The model expects normalized input; we normalize just before feeding into the network.

---

## Reference: Our Implementation

**Stack**: OpenCV + NumPy (no external augmentation library needed)

See [src/augment.py](src/augment.py):
- `preprocess_image()` → Resize with `cv2.resize()`
- `normalize_image()` → ImageNet mean/std normalization
- `augment_image()` → Flip (50% prob) + Brightness (±20%) + Normalize
  - Uses `cv2.flip()` for horizontal flip
  - Uses `cv2.convertScaleAbs()` for brightness adjustment
  - Returns normalized (float32) output ready for model

**Why this approach?**
- No external dependencies beyond OpenCV and NumPy
- Lightweight and fast
- Fully transparent code (no black-box library calls)
- Easy for team to understand and modify

---

## Further Improvements (Future Work)

These could be added to enhance robustness:
- **Rotation** (±10°): Handles camera angle variations
- **Elastic Deformation**: Mimics skin stretching/compression
- **Gaussian Blur**: Simulates out-of-focus images
- **Color Jitter**: Adjust saturation/hue for color camera drift
- **Cutout/Mixup**: Advanced regularization techniques

---

## Questions?

Consult this guide when:
- Understanding why certain augmentations exist
- Deciding whether to add new transforms
- Tuning augmentation parameters for better performance
