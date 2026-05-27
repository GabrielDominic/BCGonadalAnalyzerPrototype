"""
Full Pipeline: Stain Normalization + Preprocessing + Deep Learning
Tegillarca granosa (Blood Cockle) Gonadal Stage Classification

Model  : DenseNet-121 with Transfer Learning
Classes: Developing | Mature | Spawning | Spent

Augmentation strategy (guaranteed safe):
  1. Dataset is split into train / val / test FIRST
  2. Offline augmentation runs on training indices ONLY
     → saved to dataset_augmented_train/ (separate folder)
     → each minority class is augmented to match the largest class
  3. Val and test sets are NEVER augmented — always original images
  4. On-the-fly train_transforms add extra variation each epoch
  5. WeightedRandomSampler balances class sampling during training
"""

import os
import cv2
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import DenseNet121_Weights
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "raw_data_dir":    "./dataset_raw",
    "norm_data_dir":   "./dataset_normalized",
    "prep_data_dir":   "./dataset_prepared",
    "aug_train_dir":   "./dataset_augmented_train",
    "reference_image": "./reference.jpg",
    "image_size":      224,
    "batch_size":      8,
    "num_epochs":      20,
    "learning_rate":   0.0001,
    "num_classes":     4,
    "train_split":     0.70,
    "val_split":       0.15,
    "test_split":      0.15,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "seed":            42,
    "class_names":     ["developing", "mature", "spawning", "spent"],
    "fine_tune_epoch": 10,
}

print(f"Using device: {CONFIG['device']}")
torch.manual_seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ─────────────────────────────────────────────
# 2. REINHARD STAIN NORMALIZATION
# ─────────────────────────────────────────────
def get_mean_and_std(image_lab):
    x_mean, x_std = cv2.meanStdDev(image_lab)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std  = np.hstack(np.around(x_std,  2))
    return x_mean, x_std


def reinhard_normalize(input_img_bgr, ref_mean, ref_std):
    img_lab  = cv2.cvtColor(input_img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    inp_mean, inp_std = get_mean_and_std(img_lab.astype(np.uint8))
    inp_mean = inp_mean.reshape(1, 1, 3)
    inp_std  = inp_std.reshape(1, 1, 3)
    normalized = (img_lab - inp_mean) * (ref_std / (inp_std + 1e-6)) + ref_mean
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)


def run_reinhard_on_dataset(raw_dir, norm_dir, reference_path, class_names):
    """Run once on raw images. Comment out in main after first run."""
    print("\n" + "=" * 55)
    print("STEP 1: Reinhard Stain Normalization")
    print("=" * 55)
    ref_img = cv2.imread(reference_path)
    if ref_img is None:
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    ref_lab  = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_mean, ref_std = get_mean_and_std(ref_lab.astype(np.uint8))
    ref_mean = ref_mean.reshape(1, 1, 3)
    ref_std  = ref_std.reshape(1, 1, 3)
    for class_name in class_names:
        in_dir  = os.path.join(raw_dir,  class_name)
        out_dir = os.path.join(norm_dir, class_name)
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(in_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))]
        print(f"  [{class_name}] Normalizing {len(files)} images...")
        for fname in files:
            img = cv2.imread(os.path.join(in_dir, fname))
            if img is None:
                continue
            cv2.imwrite(os.path.join(out_dir, fname),
                        reinhard_normalize(img, ref_mean, ref_std))
    print("  Stain normalization complete.\n")


# ─────────────────────────────────────────────
# 3. IMAGE PREPROCESSING & SEGMENTATION
# ─────────────────────────────────────────────
def preprocess_image_segmented(img):
    """
    Full preprocessing for one BGR image:
      1. Illumination correction (Gaussian blur on L channel)
      2. Bilateral filter (noise reduction, edge-preserving)
      3. CLAHE on L channel (local contrast enhancement)
      4. Otsu threshold on A channel → tissue mask
      5. Morphological open + close to clean mask
      6. Bounding-box crop of tissue region
      7. Resize to 256×256
    Returns RGB numpy array.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_blur = cv2.GaussianBlur(L, (99, 99), 0)
    cv2.divide(lab[:, :, 0], L_blur, scale=255)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.bilateralFilter(img, 7, 40, 40)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L   = clahe.apply(L)
    img = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
    lab2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab2)
    A_blur  = cv2.GaussianBlur(A, (9, 9), 0)
    _, mask = cv2.threshold(A_blur, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)
    x, y, w, h = cv2.boundingRect(coords)
    img_crop   = img[y:y+h, x:x+w]
    img_crop   = cv2.resize(img_crop, (256, 256))
    return cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)


def run_preprocessing_on_dataset(norm_dir, prep_dir, class_names):
    """Run once after normalization. Comment out in main after first run."""
    print("=" * 55)
    print("STEP 2: Image Preprocessing & Segmentation")
    print("=" * 55)
    for class_name in class_names:
        in_dir  = os.path.join(norm_dir, class_name)
        out_dir = os.path.join(prep_dir, class_name)
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(in_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))]
        print(f"  [{class_name}] Preprocessing {len(files)} images...")
        for fname in files:
            img = cv2.imread(os.path.join(in_dir, fname))
            if img is None:
                continue
            preprocessed = preprocess_image_segmented(img)
            cv2.imwrite(os.path.join(out_dir, fname),
                        cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
    print("  Preprocessing complete.\n")


# ─────────────────────────────────────────────
# 4. OFFLINE AUGMENTATION — TRAINING SET ONLY
# ─────────────────────────────────────────────
def augment_image(img_rgb):
    """
    Apply a random combination of augmentations to one RGB image.
    Each transform fires with 50% probability — every generated
    image looks different from the source.
    """
    img = img_rgb.copy()
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.5:
        img = cv2.flip(img, 0)
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        h, w  = img.shape[:2]
        M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2)
        beta  = random.randint(-20, 20)
        img   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        img   = cv2.GaussianBlur(img, (ksize, ksize), 0)
    if random.random() > 0.5:
        h, w     = img.shape[:2]
        strength = random.uniform(3, 6)
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w).astype(np.float32) * 2 - 1),
            (15, 15), 0) * strength
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w).astype(np.float32) * 2 - 1),
            (15, 15), 0) * strength
        x_map = (np.meshgrid(np.arange(w), np.arange(h))[0] + dx).astype(np.float32)
        y_map = (np.meshgrid(np.arange(w), np.arange(h))[1] + dy).astype(np.float32)
        img   = cv2.remap(img, x_map, y_map,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    if random.random() > 0.5:
        h, w      = img.shape[:2]
        crop_frac = random.uniform(0.80, 0.95)
        ch, cw    = int(h * crop_frac), int(w * crop_frac)
        top       = random.randint(0, h - ch)
        left      = random.randint(0, w - cw)
        img       = img[top:top+ch, left:left+cw]
        img       = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def augment_train_only(base_dataset, train_indices, class_names, output_dir):
    """
    Safe offline augmentation — runs AFTER the train/val/test split.

    Receives only the training indices. Val and test indices are never
    passed here so they cannot be touched.

    Process:
      1. Copies all original training images to output_dir
      2. Counts training images per class
      3. Detects the largest training class automatically
      4. Generates augmented copies for minority classes until every
         class matches the largest class count
      5. Saves aug_ prefixed files alongside originals in output_dir

    After this function runs, output_dir contains a perfectly balanced
    training dataset where every class has the same number of images.
    """
    print("=" * 55)
    print("STEP 3: Train-Only Offline Augmentation")
    print("=" * 55)
    print("  Train/val/test split already done.")
    print("  Only training indices are used here.\n")

    # Group training indices by class
    class_indices = {i: [] for i in range(len(class_names))}
    for idx in train_indices:
        _, label = base_dataset.samples[idx]
        class_indices[label].append(idx)

    # Count per class in training split
    class_counts = {class_names[i]: len(v) for i, v in class_indices.items()}

    print("  Training images per class (from split):")
    for cls, cnt in class_counts.items():
        bar = "█" * (cnt // 3)
        print(f"    {cls:<14}: {cnt:>4}  {bar}")

    # Auto-detect largest class as target
    target_count = max(class_counts.values())
    largest_cls  = max(class_counts, key=class_counts.get)
    print(f"\n  Largest class  : '{largest_cls}' with {target_count} images")
    print(f"  All classes will be augmented to match: {target_count} images\n")

    # Rebuild output_dir from scratch each run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    # Copy originals
    print("  Copying original training images...")
    for label, class_name in enumerate(class_names):
        copied = 0
        for idx in class_indices[label]:
            img_path, _ = base_dataset.samples[idx]
            dst_path    = os.path.join(output_dir, class_name,
                                       os.path.basename(img_path))
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                cv2.imwrite(dst_path, img_bgr)
                copied += 1
        print(f"    [{class_name}] Copied {copied} original images.")

    # Generate augmented images for minority classes
    print()
    for label, class_name in enumerate(class_names):
        current = class_counts[class_name]
        needed  = target_count - current

        if needed <= 0:
            print(f"  [{class_name}] At target already "
                  f"({current} images). Skipping.")
            continue

        print(f"  [{class_name}] Generating {needed} augmented images "
              f"({current} → {target_count})...")

        source_indices = class_indices[label]
        generated      = 0
        while generated < needed:
            src_idx     = random.choice(source_indices)
            src_path, _ = base_dataset.samples[src_idx]
            img_bgr     = cv2.imread(src_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            aug_rgb = augment_image(img_rgb)
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
            base_name, ext = os.path.splitext(os.path.basename(src_path))
            out_fname      = f"aug_{base_name}_{generated:04d}{ext}"
            cv2.imwrite(os.path.join(output_dir, class_name, out_fname),
                        aug_bgr)
            generated += 1
        print(f"    Done — {generated} augmented images saved.")

    # Final summary
    print("\n  ── Final training dataset ──────────────────────")
    grand_total = 0
    for class_name in class_names:
        class_dir  = os.path.join(output_dir, class_name)
        all_files  = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        originals  = [f for f in all_files if not f.startswith('aug_')]
        augmented  = [f for f in all_files if f.startswith('aug_')]
        total      = len(all_files)
        grand_total += total
        bar = "█" * (total // 3)
        print(f"    {class_name:<14}: {total:>4}  "
              f"({len(originals)} orig + {len(augmented)} aug)  {bar}")
    print(f"    {'TOTAL':<14}: {grand_total:>4} training images")
    print(f"\n  Saved to   : {output_dir}")
    print(f"  Val / test : untouched — original images only\n")


# ─────────────────────────────────────────────
# 5. DATASET CLASSES
# ─────────────────────────────────────────────
class GonadalDataset(Dataset):
    """
    Loads all image paths and labels from a class-folder directory.
    No transform is applied here — transforms are handled by
    SplitDataset so each split gets its own independent transform.
    """
    def __init__(self, data_dir, class_names):
        self.samples = []
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        return Image.open(img_path).convert("RGB"), label


class SplitDataset(Dataset):
    """
    Wraps a GonadalDataset with a list of indices and an independent
    transform. This guarantees augmentation isolation between splits.

    Each split (train / val / test) gets its own SplitDataset with
    its own transform — there is no shared state between splits.

    Without this class, PyTorch's random_split() subsets all share
    the same underlying dataset object. Setting .dataset.transform
    on one subset changes it for all three simultaneously.
    """
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices      = indices
        self.transform    = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────
# 6. TRANSFORMS
# ─────────────────────────────────────────────
# train_transforms — on-the-fly augmentation applied each epoch
# on top of the already-augmented images in aug_train_dir.
# Applied only to training images via SplitDataset.
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

# val_test_transforms — no augmentation. Resize, tensor, normalize only.
# Applied to validation and test images.
val_test_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 7. SPLIT DATASET
# ─────────────────────────────────────────────
def split_dataset(prep_dir, class_names):
    """
    Load the preprocessed dataset and compute train / val / test indices.
    Returns the base dataset and the three index lists.
    Augmentation runs on training indices only in the next step.
    """
    base_dataset = GonadalDataset(prep_dir, class_names)
    total        = len(base_dataset)
    train_size   = int(CONFIG["train_split"] * total)
    val_size     = int(CONFIG["val_split"]   * total)
    test_size    = total - train_size - val_size

    all_indices   = torch.randperm(
        total, generator=torch.Generator().manual_seed(CONFIG["seed"])
    ).tolist()
    train_indices = all_indices[:train_size]
    val_indices   = all_indices[train_size:train_size + val_size]
    test_indices  = all_indices[train_size + val_size:]

    print("=" * 55)
    print("Dataset Split")
    print("=" * 55)
    print(f"  Total original images : {total}")
    print(f"  Train                 : {train_size}  (70%)")
    print(f"  Val                   : {val_size}   (15%) ← original only")
    print(f"  Test                  : {test_size}   (15%) ← original only\n")

    return base_dataset, train_indices, val_indices, test_indices


# ─────────────────────────────────────────────
# 8. BUILD DATALOADERS
# ─────────────────────────────────────────────
def build_loaders(base_dataset, val_indices, test_indices,
                  aug_train_dir, class_names):
    """
    Build DataLoaders after augmentation is complete.

    Train loader:
      - Reads from aug_train_dir (balanced: originals + aug_ files)
      - Applies train_transforms (on-the-fly augmentation each epoch)
      - WeightedRandomSampler for balanced class sampling

    Val and test loaders:
      - Read from base_dataset using original val/test indices only
      - Apply val_test_transforms (no augmentation ever)
      - No resampling
    """
    # Training — load from balanced augmented folder
    aug_dataset   = GonadalDataset(aug_train_dir, class_names)
    all_aug_idx   = list(range(len(aug_dataset)))
    train_dataset = SplitDataset(aug_dataset, all_aug_idx,
                                 transform=train_transforms)

    # Val and test — original images only, no augmentation
    val_dataset  = SplitDataset(base_dataset, val_indices,
                                transform=val_test_transforms)
    test_dataset = SplitDataset(base_dataset, test_indices,
                                transform=val_test_transforms)

    # WeightedRandomSampler
    train_labels   = [aug_dataset.samples[i][1] for i in all_aug_idx]
    class_counts   = np.bincount(train_labels, minlength=len(class_names))
    class_weights  = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.tensor(
        [class_weights[lbl] for lbl in train_labels], dtype=torch.float
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights),
        replacement=True
    )

    print("=" * 55)
    print("DataLoader Summary")
    print("=" * 55)
    print(f"  Training images (aug folder) : {len(aug_dataset)}")
    print(f"  Val images (original)        : {len(val_dataset)}")
    print(f"  Test images (original)       : {len(test_dataset)}")
    print(f"\n  Training class counts (augmented):")
    for i, cls in enumerate(class_names):
        bar = "█" * (class_counts[i] // 3)
        print(f"    {cls:<14}: {class_counts[i]:>4}  {bar}")
    print(f"\n  WeightedRandomSampler    : ON  (training only)")
    print(f"  On-the-fly augmentation  : ON  (training only)")
    print(f"  Val/Test augmentation    : OFF (original images only)\n")

    train_loader = DataLoader(train_dataset,
                              batch_size=CONFIG["batch_size"],
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset,
                              batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,
                              batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 9. MODEL — DenseNet-121
# ─────────────────────────────────────────────
def build_model(num_classes, freeze_backbone=True):
    """
    Load pretrained DenseNet-121 and replace classifier head.

    Fine-tuning strategy:
      Phase 1: Train classifier only
      Phase 2: Unfreeze entire backbone
    """
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    # Freeze backbone initially
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier (DenseNet uses .classifier)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    print(f"  Model        : DenseNet-121")
    print(f"  Total params : {total:,}")
    print(f"  Trainable    : {trainable:,} (classifier — backbone frozen)\n")

    return model.to(CONFIG["device"])


# ─────────────────────────────────────────────
# 10. TRAINING & EVALUATION FUNCTIONS
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


# ─────────────────────────────────────────────
# 11. TRAIN THE MODEL
# ─────────────────────────────────────────────
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    history       = {"train_loss": [], "val_loss": [],
                     "train_acc":  [], "val_acc":  []}
    best_val_loss = float("inf")

    print("=" * 55)
    print("STEP 4: Training DenseNet-121")
    print("=" * 55)

    for epoch in range(1, CONFIG["num_epochs"] + 1):

        # Phase 2: Unfreeze deeper layers
        if epoch == CONFIG["fine_tune_epoch"]:
            print("\n>>> Unfreezing entire DenseNet backbone...\n")
            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, CONFIG["device"])
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, CONFIG["device"])
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_densenet121.pth")
            saved = "✓ Saved"

        print(f"Epoch [{epoch:02d}/{CONFIG['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} {saved}")

    print(f"\nBest model saved to: best_model_densenet121.pth")
    return history


# ─────────────────────────────────────────────
# 12. EVALUATE ON TEST SET
# ─────────────────────────────────────────────
def evaluate_on_test(model, test_loader, class_names):
    """
    Load best model and evaluate on held-out test set.
    Test images are always original — never augmented.
    """
    model.load_state_dict(torch.load("best_model_densenet121.pth",
                                     map_location=CONFIG["device"]))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(CONFIG["device"])
            outputs = model(images)
            preds   = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print("\n" + "=" * 55)
    print("TEST SET RESULTS — DenseNet-121")
    print("=" * 55)
    print(classification_report(all_labels, all_preds,
                                target_names=class_names))
    return all_labels, all_preds


# ─────────────────────────────────────────────
# 13. VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_results(history, all_labels, all_preds, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DenseNet-121 — Training Results", fontsize=13,
                 fontweight="bold")
    axes[0].plot(history["train_loss"], label="Train Loss", color="#185FA5")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#BA7517")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["train_acc"], label="Train Acc", color="#185FA5")
    axes[1].plot(history["val_acc"],   label="Val Acc",   color="#BA7517")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves_densenet121.png", dpi=150)
    plt.show()
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — DenseNet-121\nGonadal Stage Classification")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_densenet121.png", dpi=150)
    plt.show()
    print("Saved: training_curves_densenet121.png, confusion_matrix_densenet121.png")


# ─────────────────────────────────────────────
# 14. SINGLE IMAGE INFERENCE
# ─────────────────────────────────────────────
def predict_image(image_path, model, class_names, device):
    """
    Preprocess a raw image and run inference.
    Always uses val_test_transforms — no augmentation during prediction.
    Prints probability breakdown and saves a bar chart PNG.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    preprocessed_rgb = preprocess_image_segmented(img_bgr)
    pil_img  = Image.fromarray(preprocessed_rgb)
    tensor   = val_test_transforms(pil_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output   = model(tensor)
        probs    = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())
    predicted_class = class_names[pred_idx]

    print("\n" + "=" * 55)
    print(f"PREDICTION: {os.path.basename(image_path)}")
    print("=" * 55)
    print(f"  Model           : DenseNet-121")
    print(f"  Predicted stage : {predicted_class.upper()}")
    print(f"  Confidence      : {probs[pred_idx]*100:.2f}%\n")
    print("  Probability breakdown:")
    print("  " + "-" * 40)
    for i, (cls, prob) in enumerate(zip(class_names, probs)):
        bar    = "█" * int(prob * 30)
        marker = " ◄" if i == pred_idx else ""
        print(f"  {cls:<12} {prob*100:6.2f}%  {bar}{marker}")
    print("  " + "-" * 40)

    colors      = ["#9FE1CB" if i == pred_idx else "#D3D1C7"
                   for i in range(len(class_names))]
    edge_colors = ["#0F6E56" if i == pred_idx else "#B4B2A9"
                   for i in range(len(class_names))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    fig.suptitle(f"DenseNet-121 — Gonadal Stage Prediction\n"
                 f"{os.path.basename(image_path)}",
                 fontsize=12, fontweight="bold", y=1.02)
    axes[0].imshow(preprocessed_rgb)
    axes[0].set_title("Preprocessed image", fontsize=11)
    axes[0].axis("off")
    bars = axes[1].barh(class_names, probs * 100,
                        color=colors, edgecolor=edge_colors,
                        linewidth=1.2, height=0.55)
    for bar, prob in zip(bars, probs):
        p_idx = probs.tolist().index(prob)
        axes[1].text(bar.get_width() + 0.8,
                     bar.get_y() + bar.get_height() / 2,
                     f"{prob*100:.2f}%",
                     va="center", ha="left", fontsize=11,
                     fontweight="bold" if p_idx == pred_idx else "normal",
                     color="#0F6E56" if p_idx == pred_idx else "#444441")
    axes[1].set_xlim(0, 115)
    axes[1].set_xlabel("Probability (%)", fontsize=11)
    axes[1].set_title(
        f"Predicted: {predicted_class.upper()}  "
        f"({probs[pred_idx]*100:.2f}% confidence)",
        fontsize=11, color="#0F6E56", fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].tick_params(axis="y", labelsize=11)
    axes[1].tick_params(axis="x", labelsize=10)
    axes[1].invert_yaxis()
    plt.tight_layout()
    out_path = os.path.splitext(image_path)[0] + "_densenet121_prediction.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Chart saved to: {out_path}")
    return predicted_class, probs


# ─────────────────────────────────────────────
# 15. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── STEP 1: Stain Normalization ───────────────────────────────
    # Run once. Comment out after first run.
    # run_reinhard_on_dataset(
    #     raw_dir        = CONFIG["raw_data_dir"],
    #     norm_dir       = CONFIG["norm_data_dir"],
    #     reference_path = CONFIG["reference_image"],
    #     class_names    = CONFIG["class_names"],
    # )

    # # ── STEP 2: Preprocessing ─────────────────────────────────────
    # # Run once. Comment out after first run.
    # run_preprocessing_on_dataset(
    #     norm_dir    = CONFIG["norm_data_dir"],
    #     prep_dir    = CONFIG["prep_data_dir"],
    #     class_names = CONFIG["class_names"],
    # )

    # ── STEP 3: Split dataset FIRST ───────────────────────────────
    # Computes train / val / test indices from original images.
    # Augmentation will only use train_indices in the next step.
    base_dataset, train_indices, val_indices, test_indices = split_dataset(
        CONFIG["prep_data_dir"], CONFIG["class_names"]
    )

    # ── STEP 4: Augment training images only ──────────────────────
    # Uses train_indices only — val and test are completely isolated.
    # Minority classes are augmented to match the largest class count.
    # Run once. Comment out on subsequent runs to skip re-generation.
    # augment_train_only(
    #     base_dataset  = base_dataset,
    #     train_indices = train_indices,
    #     class_names   = CONFIG["class_names"],
    #     output_dir    = CONFIG["aug_train_dir"],
    # )

    # ── STEP 5: Build DataLoaders ─────────────────────────────────
    # Train loader reads from aug_train_dir (balanced).
    # Val and test loaders read from original base_dataset.
    train_loader, val_loader, test_loader = build_loaders(
        base_dataset  = base_dataset,
        val_indices   = val_indices,
        test_indices  = test_indices,
        aug_train_dir = CONFIG["aug_train_dir"],
        class_names   = CONFIG["class_names"],
    )

    # ── STEP 6: Build Model ───────────────────────────────────────
    model = build_model(CONFIG["num_classes"], freeze_backbone=True)

    # ── STEP 7: Train ─────────────────────────────────────────────
    history = train_model(model, train_loader, val_loader)

    # ── STEP 8: Evaluate on Test Set ──────────────────────────────
    all_labels, all_preds = evaluate_on_test(
        model, test_loader, CONFIG["class_names"]
    )
    plot_results(history, all_labels, all_preds, CONFIG["class_names"])

    # ── STEP 9: Single Image Prediction ───────────────────────────
    # model.load_state_dict(torch.load("best_model_densenet121.pth"))
    # predict_image(
    #     image_path  = "sample.jpg",
    #     model       = model,
    #     class_names = CONFIG["class_names"],
    #     device      = CONFIG["device"],
    # )
