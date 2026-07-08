# -*- coding: utf-8 -*-
"""
freqfss_isic2016.py
════════════════════════════════════════════════════════════════════════════════
FreqFSS: Frequency-Guided Prototype Alignment for Few-Shot MRI Segmentation
         — adapted for ISIC 2016 skin lesion segmentation

ARCHITECTURE OVERVIEW
─────────────────────
  Dual-pathway encoder (shared weights across support/query):
    • Spatial encoder   : ResNet-50 backbone  → F_s  (H×W×C)
    • Frequency encoder : 2D FFT → magnitude/phase → Spectral MLP → F_f (H×W×C)

  5-shot dual prototype construction (masked average pooling):
    • P_s  : spatial prototype   averaged over 5 support images
    • P_f  : frequency prototype averaged over 5 support images

  Adaptive Band Weighting (ABW):
    • Decomposes frequency features into 3 radial rings (low/mid/high)
    • MLP conditioned on P_s predicts per-band scalar weights

  Cross-Domain Alignment Module (CDAM) — the core novelty:
    • Bidirectional cross-attention: P_s ↔ P_f → P_fused
    • Dual similarity maps: S_spatial (P_fused · Q_s) + S_freq (P_f · Q_f)
    • Learnable α,β fusion

  Prototype-guided UNet decoder:
    • S injected at bottleneck via concatenation
    • 3 skip connections from encoder stages (E1, E2, E3)
    • Output: 256×256×1 sigmoid probability mask

TRAINING PROTOCOL
─────────────────
  • Episodic training (PANet-style): N_way=1, K_shot=5
  • ISIC 2016 is single-class (lesion vs background), so N_way=1
  • Each episode: 5 support image+mask pairs → prototype → 1 query prediction
  • Loss: Dice + BCE (supervised seg) + L_freq (spectral consistency) + L_align
  • Epochs: 100, cosine LR schedule, AdamW
  • Evaluation: 1000 test episodes, mean Dice / IoU / Sensitivity / Specificity

DATA LOADING (mirrors your existing ISIC 2016 code exactly)
─────────────────────────────────────────────────────────────
  Dataset  : jotiradityabanerjee/segmentation-isic-ham10k
  Structure: ISIC16_Resized256x256/{Train,Test}/{Images,Masks}/
  Mask name: {base}_Segmentation.png
  Reading  : cv2 BGR→RGB, masks binarized at 127
  Norm     : ImageNet mean/std, 256×256

HOW TO RUN IN KAGGLE
─────────────────────
  1. Attach dataset jotiradityabanerjee/segmentation-isic-ham10k
  2. Paste entire file into one cell and run, OR:
       results = train_freqfss()          # full training + evaluation
       results = train_freqfss(epochs=10) # quick sanity check
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import copy
import json
import math
import random
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Paths ────────────────────────────────────────────────────────
    DATA_PATH         = os.environ.get(
        'FREQFSS_DATA',
        '/kaggle/input/datasets/jotiradityabanerjee/segmentation-isic-ham10k'  # Exactly matches the screenshot
    )
    
    # Target the ISIC 2016 folder as per your script's docstring
    SEG_PATH          = os.path.join(DATA_PATH, 'ISIC16_Resized256x256') 
    
    TRAIN_IMG_DIR     = os.path.join(SEG_PATH, 'Train', 'Images')
    TRAIN_MASK_DIR    = os.path.join(SEG_PATH, 'Train', 'Masks')
    TEST_IMG_DIR      = os.path.join(SEG_PATH, 'Test',  'Images')
    TEST_MASK_DIR     = os.path.join(SEG_PATH, 'Test',  'Masks')
    SAVE_DIR          = './freqfss_results'

    # ── Image ─────────────────────────────────────────────────────────────────
    IMG_SIZE    = 256
    NUM_CLASSES = 1       # binary: lesion vs background

    # ── Episode protocol ──────────────────────────────────────────────────────
    N_WAY    = 1          # single-class binary segmentation
    K_SHOT   = 5          # 5 support images per episode
    N_QUERY  = 1          # queries per episode during training

    # ── Split ratios (same as your existing code) ─────────────────────────────
    VAL_RATIO  = 0.15     # of train split
    TEST_RATIO = 0.20     # held-out test images

    # ── Training ──────────────────────────────────────────────────────────────
    EPOCHS          = 100
    EPISODES_TRAIN  = 200   # episodes per epoch
    EPISODES_VAL    = 200   # episodes for validation
    EPISODES_TEST   = 1000  # episodes for final test evaluation
    LR              = 1e-4
    LR_MIN          = 1e-6
    WEIGHT_DECAY    = 1e-4
    BATCH_SIZE      = 1     # 1 episode per forward pass (memory safety on T4/P100)
    NUM_WORKERS     = 0
    PIN_MEMORY      = False

    # ── Loss weights ──────────────────────────────────────────────────────────
    LAMBDA_FREQ  = 0.3    # spectral consistency loss weight
    LAMBDA_ALIGN = 0.2    # prototype alignment loss weight

    # ── Architecture ──────────────────────────────────────────────────────────
    FEAT_DIM      = 256   # channel dim of spatial/freq features
    FREQ_BANDS    = 3     # low / mid / high radial frequency rings
    ATTN_HEADS    = 4     # cross-attention heads in CDAM
    PRETRAINED    = True  # ImageNet-pretrained ResNet-50

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

# ── 2.1  Augmentation pipelines ───────────────────────────────────────────────

_MEAN = (0.485, 0.456, 0.406)   # ImageNet (same as your existing code)
_STD  = (0.229, 0.224, 0.225)


def support_transform(img_size: int) -> A.Compose:
    """Augmentation for support images — moderate, preserves mask alignment."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def query_transform(img_size: int) -> A.Compose:
    """Augmentation for query images — stronger to improve generalization."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0), p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08, p=0.6),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def val_transform(img_size: int) -> A.Compose:
    """No augmentation for validation / test queries."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


# ── 2.2  Index builder (identical logic to your existing code) ────────────────

def build_isic_index(config: Config):
    """
    Scans Train/ and Test/ directories for matched image-mask pairs.
    Mask naming convention: {base}_Segmentation.png  (from your existing code).
    Returns: img_paths (list[str]), mask_paths (list[str])
    """
    img_paths, mask_paths = [], []
    for img_dir, mask_dir in [
        (config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR),
        (config.TEST_IMG_DIR,  config.TEST_MASK_DIR),
    ]:
        if not os.path.exists(img_dir):
            print(f"  ⚠ Directory not found: {img_dir}")
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            base      = os.path.splitext(fname)[0]
            mask_name = base + '_Segmentation.png'
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                img_paths.append(os.path.join(img_dir, fname))
                mask_paths.append(mask_path)

    print(f"  ✓ Total paired samples found: {len(img_paths)}")
    return img_paths, mask_paths


def split_dataset(img_paths, mask_paths, config: Config, seed: int = 42):
    """
    Splits the full dataset into train / val / test subsets.
    Ratios mirror your existing code: 20% test, then 15% of rest for val.
    Returns: (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)
    """
    n       = len(img_paths)
    indices = list(range(n))
    rng     = random.Random(seed)
    rng.shuffle(indices)

    n_test   = max(1, int(config.TEST_RATIO * n))
    test_idx = indices[-n_test:]
    rest_idx = indices[:-n_test]

    n_val    = max(1, int(config.VAL_RATIO * len(rest_idx)))
    val_idx  = rest_idx[:n_val]
    train_idx = rest_idx[n_val:]

    def subset(idx_list):
        imgs  = [img_paths[i]  for i in idx_list]
        masks = [mask_paths[i] for i in idx_list]
        return imgs, masks

    print(f"  Train: {len(train_idx):>5}  |  Val: {len(val_idx):>4}  |  Test: {len(test_idx):>4}")
    return subset(train_idx), subset(val_idx), subset(test_idx)


# ── 2.3  Single image/mask loader (reusable, follows your cv2 pattern) ────────

def load_image_mask(img_path: str, mask_path: str):
    """
    Loads one BGR image (→ RGB) and its binary mask exactly as in your code:
      image : np.ndarray  H×W×3  uint8
      mask  : np.ndarray  H×W    float32  (0.0 or 1.0)
    """
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    mask = (mask > 127).astype(np.float32)   # same threshold as your code
    return image, mask


# ── 2.4  Episodic dataset ──────────────────────────────────────────────────────

class EpisodicISICDataset(Dataset):
    """
    Generates few-shot episodes on-the-fly.

    Each episode returns:
        support_imgs   : Tensor  (K, 3, H, W)   — K support images
        support_masks  : Tensor  (K, 1, H, W)   — K support binary masks
        query_img      : Tensor  (3, H, W)       — 1 query image
        query_mask     : Tensor  (1, H, W)       — ground truth for loss/eval

    For ISIC 2016 (binary, single class) the episodic sampling simply draws
    K+1 different images from the available pool.  The K support images provide
    the prototype; the remaining image is the query to segment.

    NOTE: Unlike multi-class few-shot datasets (e.g. Pascal-5i), ISIC 2016 is
    already a single-class binary problem — every image belongs to the "lesion"
    class.  So each episode is: sample K support images, 1 query image, all
    from the same dataset split.  This matches the standard PANet / SSL-ALPNet
    episodic binary segmentation protocol used on medical imaging datasets.
    """
    def __init__(
        self,
        img_paths,
        mask_paths,
        k_shot:     int   = 5,
        n_episodes: int   = 1000,
        img_size:   int   = 256,
        mode:       str   = 'train',   # 'train' | 'val' | 'test'
        seed:       int   = 42,
    ):
        assert len(img_paths) == len(mask_paths), "img and mask lists must match"
        assert len(img_paths) >= k_shot + 1,      \
            f"Need at least K+1={k_shot+1} images; got {len(img_paths)}"

        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.k_shot     = k_shot
        self.n_episodes = n_episodes
        self.img_size   = img_size
        self.mode       = mode

        # Pre-generate episode indices for reproducibility
        rng = random.Random(seed)
        self.episodes = []
        n = len(img_paths)
        for _ in range(n_episodes):
            indices = rng.sample(range(n), k_shot + 1)
            support_idx = indices[:k_shot]
            query_idx   = indices[k_shot]
            self.episodes.append((support_idx, query_idx))

        # Transforms
        if mode == 'train':
            self.sup_tf   = support_transform(img_size)
            self.query_tf = query_transform(img_size)
        else:
            self.sup_tf   = val_transform(img_size)
            self.query_tf = val_transform(img_size)

    def __len__(self):
        return self.n_episodes

    def _load_and_transform(self, img_path, mask_path, transform):
        image, mask = load_image_mask(img_path, mask_path)
        aug   = transform(image=image, mask=mask)
        img_t = aug['image']                           # (3,H,W) float32 tensor
        msk_t = aug['mask'].unsqueeze(0).float()       # (1,H,W)
        return img_t, msk_t

    def __getitem__(self, idx):
        support_indices, query_idx = self.episodes[idx]

        # ── Support images & masks ──────────────────────────────────────────
        sup_imgs, sup_masks = [], []
        for s_idx in support_indices:
            img_t, msk_t = self._load_and_transform(
                self.img_paths[s_idx],
                self.mask_paths[s_idx],
                self.sup_tf,
            )
            sup_imgs.append(img_t)
            sup_masks.append(msk_t)

        support_imgs  = torch.stack(sup_imgs,  dim=0)  # (K,3,H,W)
        support_masks = torch.stack(sup_masks, dim=0)  # (K,1,H,W)

        # ── Query image & mask ─────────────────────────────────────────────
        query_img, query_mask = self._load_and_transform(
            self.img_paths[query_idx],
            self.mask_paths[query_idx],
            self.query_tf,
        )

        return support_imgs, support_masks, query_img, query_mask


def make_episode_loaders(config: Config):
    """
    Builds train / val / test EpisodicISICDataset instances and their loaders.
    Uses the same dataset index + split logic as your existing code.
    """
    img_paths, mask_paths = build_isic_index(config)
    (tr_imgs, tr_masks), (va_imgs, va_masks), (te_imgs, te_masks) = \
        split_dataset(img_paths, mask_paths, config, seed=config.SEED)

    loader_kw = dict(num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    train_ds = EpisodicISICDataset(
        tr_imgs, tr_masks,
        k_shot=config.K_SHOT,
        n_episodes=config.EPISODES_TRAIN,
        img_size=config.IMG_SIZE,
        mode='train', seed=config.SEED,
    )
    val_ds = EpisodicISICDataset(
        va_imgs, va_masks,
        k_shot=config.K_SHOT,
        n_episodes=config.EPISODES_VAL,
        img_size=config.IMG_SIZE,
        mode='val', seed=config.SEED + 1,
    )
    test_ds = EpisodicISICDataset(
        te_imgs, te_masks,
        k_shot=config.K_SHOT,
        n_episodes=config.EPISODES_TEST,
        img_size=config.IMG_SIZE,
        mode='test', seed=config.SEED + 2,
    )

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, **loader_kw)

    print(f"  Train episodes : {len(train_ds)}")
    print(f"  Val   episodes : {len(val_ds)}")
    print(f"  Test  episodes : {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

# ── 3.1  Spatial Encoder (ResNet-50 backbone, returns multi-scale features) ───

class SpatialEncoder(nn.Module):
    """
    ResNet-50 encoder that returns 4 hierarchical feature maps for skip
    connections AND the bottleneck feature.

    Output channels:
        s0 : (B, 64,  H/2,  W/2)   after initial conv+BN+ReLU
        s1 : (B, 256, H/4,  W/4)   layer1  (res block, 256ch)
        s2 : (B, 512, H/8,  W/8)   layer2
        s3 : (B, 1024,H/16, W/16)  layer3
        s4 : (B, 2048,H/32, W/32)  layer4  — bottleneck input

    All weights shared between support and query forward passes.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1   # 256ch
        self.enc2 = backbone.layer2   # 512ch
        self.enc3 = backbone.layer3   # 1024ch
        self.enc4 = backbone.layer4   # 2048ch

    def forward(self, x):
        """Returns (s0, s1, s2, s3, s4) at progressively coarser resolutions."""
        s0 = self.enc0(x)             # 64ch,  H/2
        s1 = self.enc1(self.pool(s0)) # 256ch, H/4
        s2 = self.enc2(s1)            # 512ch, H/8
        s3 = self.enc3(s2)            # 1024ch,H/16
        s4 = self.enc4(s3)            # 2048ch,H/32
        return s0, s1, s2, s3, s4


# ── 3.2  Spectral MLP (maps FFT magnitude+phase → real feature map) ───────────

class SpectralMLP(nn.Module):
    """
    Converts the 2D FFT of an input image into a real-valued feature map with
    the same spatial resolution and channel dim as the spatial encoder output.

    Pipeline per image:
        1. Apply 2D FFT → complex spectrum (H×W)
        2. Split into magnitude log(1+|F|) and phase angle(F)
        3. Concatenate along channel dim → (2C_in, H, W) real tensor
        4. Feed through lightweight conv MLP → (FEAT_DIM, H, W)
        5. Bilinear resize to match spatial encoder resolution at each scale

    Design note: we compute FFT on the raw 3-channel image (C=3), giving 6
    real channels (3 mag + 3 phase) as input to the MLP.  This is computationally
    cheap and avoids coupling with the spatial encoder.
    """
    def __init__(self, in_channels: int = 6, feat_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, 64,       kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128,               kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_dim,         kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x : (B, 3, H, W)  normalized image tensor
        Returns : (B, feat_dim, H, W)
        """
        # 2D FFT per channel (operates on last two dims)
        x_fft = torch.fft.fft2(x, norm='ortho')          # complex (B,3,H,W)

        # Magnitude: log-compress to stabilize dynamic range
        magnitude = torch.log1p(torch.abs(x_fft))         # (B,3,H,W)

        # Phase: shift FFT origin to center (fftshift analog via roll)
        phase     = torch.angle(x_fft)                    # (B,3,H,W) in [-π,π]

        # Concatenate → (B,6,H,W) real-valued input to MLP
        freq_input = torch.cat([magnitude, phase], dim=1)

        return self.mlp(freq_input)                        # (B, feat_dim, H, W)


# ── 3.3  Adaptive Band Weighting (ABW) ────────────────────────────────────────

class AdaptiveBandWeighting(nn.Module):
    """
    Decomposes frequency features into 3 radial rings (low/mid/high spatial
    frequencies) and predicts per-band scalar weights conditioned on the spatial
    prototype P_s.

    Research motivation:
        Different anatomical structures occupy different frequency bands.
        A global lesion boundary → strong mid-frequency signal.
        Fine texture within the lesion → high frequency.
        Global intensity field → low frequency.
        The model learns per-episode which bands to trust.

    Implementation:
        • Build 3 binary masks in the H×W FFT grid defining radial rings
        • Pool frequency features within each ring → 3 ring descriptors
        • Concatenate with P_s → MLP → softmax → (w_low, w_mid, w_high)
        • Return weighted sum of ring-masked frequency feature map
    """
    def __init__(self, feat_dim: int = 256, img_size: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.img_size = img_size

        # MLP: [P_s (feat_dim) || ring_descriptors (3*feat_dim)] → 3 weights
        self.band_mlp = nn.Sequential(
            nn.Linear(feat_dim + 3 * feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

        # Build radial ring masks (fixed, registered as buffer)
        self.register_buffer('ring_masks', self._build_ring_masks(img_size))

    @staticmethod
    def _build_ring_masks(img_size: int) -> torch.Tensor:
        """
        Returns (3, img_size, img_size) binary float32 masks defining:
            ring 0 (low freq)  : radius 0 to img_size/6
            ring 1 (mid freq)  : radius img_size/6 to img_size/3
            ring 2 (high freq) : radius img_size/3 to img_size/2
        FFT origin assumed at center (after fftshift).
        """
        H, W = img_size, img_size
        cy, cx = H / 2.0, W / 2.0
        ys = torch.arange(H).float().unsqueeze(1).expand(H, W)
        xs = torch.arange(W).float().unsqueeze(0).expand(H, W)
        r  = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

        r_max = min(H, W) / 2.0
        masks = torch.zeros(3, H, W)
        masks[0] = (r <  r_max / 3).float()
        masks[1] = ((r >= r_max / 3) & (r < 2 * r_max / 3)).float()
        masks[2] = (r >= 2 * r_max / 3).float()
        return masks   # (3, H, W)

    def forward(self, F_f: torch.Tensor, P_s: torch.Tensor) -> torch.Tensor:
        """
        F_f : (B, feat_dim, H, W)   frequency feature map
        P_s : (B, feat_dim)         spatial prototype vector
        Returns : (B, feat_dim, H, W) band-weighted frequency features
        """
        B, C, H, W = F_f.shape

        # Resize ring masks if feature map size differs from img_size
        ring_masks = self.ring_masks  # (3, H_img, W_img)
        if ring_masks.shape[1] != H or ring_masks.shape[2] != W:
            ring_masks = F.interpolate(
                ring_masks.unsqueeze(0), size=(H, W), mode='nearest'
            ).squeeze(0)

        # Pool each frequency ring → descriptor vector per ring
        ring_descs = []
        for k in range(3):
            m   = ring_masks[k].unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
            cnt = m.sum().clamp(min=1.0)
            desc = (F_f * m).sum(dim=(2, 3)) / cnt           # (B, C)
            ring_descs.append(desc)

        # Concatenate: [P_s || desc_low || desc_mid || desc_high]
        mlp_input = torch.cat([P_s] + ring_descs, dim=1)    # (B, 4C)
        weights   = torch.softmax(self.band_mlp(mlp_input), dim=1)  # (B, 3)

        # Weighted sum of ring-masked feature maps
        out = torch.zeros_like(F_f)
        for k in range(3):
            m   = ring_masks[k].unsqueeze(0).unsqueeze(0)           # (1,1,H,W)
            w_k = weights[:, k].view(B, 1, 1, 1)                    # (B,1,1,1)
            out = out + w_k * F_f * m

        return out   # (B, feat_dim, H, W)


# ── 3.4  Cross-Domain Alignment Module (CDAM) — the core contribution ─────────

class CrossDomainAlignmentModule(nn.Module):
    """
    Bidirectional cross-attention between spatial and frequency prototypes,
    producing a fused prototype and dual similarity maps.

    Given:
        P_s  : (B, C)  spatial prototype
        P_f  : (B, C)  frequency prototype (band-weighted)
        Q_s  : (B, C, H, W)  query spatial features
        Q_f  : (B, C, H, W)  query frequency features

    Steps:
        1. Cross-attention:
               P_s' = CrossAttn(Q=P_s, K=P_f, V=P_f)
               P_f' = CrossAttn(Q=P_f, K=P_s, V=P_s)
               P_fused = LayerNorm(P_s' + P_f')

        2. Cosine similarity maps:
               S_spatial(i,j) = cosine(P_fused, Q_s[:,i,j])   shape (B,1,H,W)
               S_freq(i,j)    = cosine(P_f',   Q_f[:,i,j])   shape (B,1,H,W)

        3. Learnable weighted fusion:
               S = sigmoid(alpha) * S_spatial + sigmoid(beta) * S_freq

    The dual similarity map S is passed to the UNet decoder as additional
    spatial guidance injected at the bottleneck.
    """
    def __init__(self, feat_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.feat_dim = feat_dim

        # Project prototypes to Q/K/V spaces (multi-head cross-attention)
        self.q_proj_s = nn.Linear(feat_dim, feat_dim)
        self.k_proj_s = nn.Linear(feat_dim, feat_dim)
        self.v_proj_s = nn.Linear(feat_dim, feat_dim)

        self.q_proj_f = nn.Linear(feat_dim, feat_dim)
        self.k_proj_f = nn.Linear(feat_dim, feat_dim)
        self.v_proj_f = nn.Linear(feat_dim, feat_dim)

        self.n_heads  = n_heads
        self.head_dim = feat_dim // n_heads
        self.scale    = math.sqrt(self.head_dim)

        self.norm_s = nn.LayerNorm(feat_dim)
        self.norm_f = nn.LayerNorm(feat_dim)

        # Learnable fusion weights (initialized at 0 → sigmoid(0)=0.5 equal blend)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta  = nn.Parameter(torch.zeros(1))

    def _cross_attn(self, q_vec, k_vec, v_vec,
                    q_proj, k_proj, v_proj):
        """
        Single-step cross-attention:
          q_vec : (B, C)  — query prototype
          k_vec : (B, C)  — key prototype
        Returns: attended vector (B, C)
        """
        B  = q_vec.size(0)
        H  = self.n_heads
        Hd = self.head_dim

        Q = q_proj(q_vec).view(B, H, Hd)        # (B, H, Hd)
        K = k_proj(k_vec).view(B, H, Hd)
        V = v_proj(v_vec).view(B, H, Hd)

        # Scaled dot-product attention over heads (B×H sequences of length 1)
        attn = (Q * K).sum(dim=-1, keepdim=True) / self.scale   # (B, H, 1)
        attn = torch.softmax(attn, dim=1)                        # (B, H, 1)
        out  = (attn * V).view(B, -1)                           # (B, C)
        return out

    def forward(self, P_s, P_f, Q_s, Q_f):
        """
        P_s  : (B, C)
        P_f  : (B, C)
        Q_s  : (B, C, H, W)
        Q_f  : (B, C, H, W)
        Returns: S (B, 1, H, W), P_fused (B, C)
        """
        B, C, H, W = Q_s.shape

        # ── Cross-attention ──────────────────────────────────────────────────
        # P_s queries P_f (spatial attends to frequency)
        P_s_prime = self._cross_attn(P_s, P_f, P_f,
                                     self.q_proj_s, self.k_proj_s, self.v_proj_s)
        P_s_prime = self.norm_s(P_s + P_s_prime)        # residual + LN

        # P_f queries P_s (frequency attends to spatial)
        P_f_prime = self._cross_attn(P_f, P_s, P_s,
                                     self.q_proj_f, self.k_proj_f, self.v_proj_f)
        P_f_prime = self.norm_f(P_f + P_f_prime)        # residual + LN

        # Fused prototype
        P_fused = P_s_prime + P_f_prime                  # (B, C)

        # ── Dual similarity maps ─────────────────────────────────────────────
        # Reshape query features to (B, C, H*W) for batched cosine sim
        Q_s_flat = Q_s.view(B, C, -1)                   # (B, C, H*W)
        Q_f_flat = Q_f.view(B, C, -1)

        # Prototype vectors as (B, C, 1) for broadcasting
        P_fused_norm = F.normalize(P_fused.unsqueeze(-1), dim=1)   # (B, C, 1)
        P_f_norm     = F.normalize(P_f_prime.unsqueeze(-1), dim=1)

        Q_s_norm = F.normalize(Q_s_flat, dim=1)                    # (B, C, H*W)
        Q_f_norm = F.normalize(Q_f_flat, dim=1)

        S_spatial = (P_fused_norm * Q_s_norm).sum(dim=1)           # (B, H*W)
        S_freq    = (P_f_norm    * Q_f_norm).sum(dim=1)

        S_spatial = S_spatial.view(B, 1, H, W)
        S_freq    = S_freq.view(B, 1, H, W)

        # ── Learnable weighted fusion ────────────────────────────────────────
        a = torch.sigmoid(self.alpha)
        b = torch.sigmoid(self.beta)
        S = a * S_spatial + b * S_freq                  # (B, 1, H, W)

        return S, P_fused, P_f_prime


# ── 3.5  Prototype construction (masked average pooling over K shots) ──────────

def build_prototypes(F_s_list, F_f_list, mask_list):
    """
    Computes dual prototypes by masked average pooling over K support images.

    Args:
        F_s_list : list of K tensors, each (1, C, H, W)  — spatial features
        F_f_list : list of K tensors, each (1, C, H, W)  — frequency features
        mask_list: list of K tensors, each (1, 1, H, W)  — binary masks

    Returns:
        P_s : (1, C)  spatial prototype
        P_f : (1, C)  frequency prototype
    """
    ps_list, pf_list = [], []
    for F_s, F_f, M in zip(F_s_list, F_f_list, mask_list):
        # Resize mask to match feature map resolution
        H, W   = F_s.shape[2], F_s.shape[3]
        M_down = F.interpolate(M.float(), size=(H, W), mode='bilinear',
                               align_corners=False)
        M_down = (M_down > 0.5).float()

        denom = M_down.sum(dim=(2, 3)).clamp(min=1.0)   # (1,1)

        p_s = (F_s * M_down).sum(dim=(2, 3)) / denom    # (1, C)
        p_f = (F_f * M_down).sum(dim=(2, 3)) / denom

        ps_list.append(p_s)
        pf_list.append(p_f)

    # Average prototypes across K shots
    P_s = torch.stack(ps_list, dim=0).mean(dim=0)       # (1, C)
    P_f = torch.stack(pf_list, dim=0).mean(dim=0)

    return P_s, P_f


# ── 3.6  UNet Decoder ──────────────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, pad=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample + skip-concatenate + DoubleConv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class BottleneckBlock(nn.Module):
    """
    Bottleneck: receives encoder s4 (2048ch) concatenated with similarity map S
    (1ch) → reduces to 1024ch.

    The similarity map injection at the bottleneck is the key architectural
    decision that conditions the entire decoding path on prototype similarity.
    """
    def __init__(self, in_ch: int = 2049, out_ch: int = 1024):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, s4, S):
        """
        s4 : (B, 2048, H/32, W/32)
        S  : (B, 1,    H/?,  W/?)   — similarity map at any resolution
        """
        # Resize S to match s4 spatial dimensions
        S_resized = F.interpolate(S, size=s4.shape[2:], mode='bilinear',
                                  align_corners=False)
        x = torch.cat([s4, S_resized], dim=1)   # (B, 2049, H/32, W/32)
        return self.conv(x)                      # (B, 1024, H/32, W/32)


class FreqFSSDecoder(nn.Module):
    """
    Prototype-guided UNet decoder.

    Inputs:
        s0  : (B, 64,   H/2,  W/2)   encoder skip E1
        s1  : (B, 256,  H/4,  W/4)   encoder skip E2
        s2  : (B, 512,  H/8,  W/8)   encoder skip E3
        s3  : (B, 1024, H/16, W/16)  encoder skip E4 (used in bottleneck upblock)
        s4  : (B, 2048, H/32, W/32)  bottleneck input
        S   : (B, 1,    H,    W)     similarity map

    Output:
        (B, 1, H, W)  sigmoid probability mask
    """
    def __init__(self):
        super().__init__()
        self.bottleneck = BottleneckBlock(in_ch=2049, out_ch=1024)

        # up4: 1024 → 512, skip s3=1024 → out 512
        self.up4 = UpBlock(1024, 1024, 512)
        # up3: 512 → 256, skip s2=512 → out 256
        self.up3 = UpBlock(512,  512,  256)
        # up2: 256 → 128, skip s1=256 → out 128
        self.up2 = UpBlock(256,  256,  128)
        # up1: 128 → 64,  skip s0=64  → out 64
        self.up1 = UpBlock(128,  64,   64)

        # Final 2× upsample + 1×1 conv → binary logit
        self.final_up   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, s0, s1, s2, s3, s4, S):
        b  = self.bottleneck(s4, S)   # (B,1024,H/32,W/32)
        d4 = self.up4(b,  s3)         # (B,512, H/16,W/16)
        d3 = self.up3(d4, s2)         # (B,256, H/8, W/8)
        d2 = self.up2(d3, s1)         # (B,128, H/4, W/4)
        d1 = self.up1(d2, s0)         # (B,64,  H/2, W/2)
        out = self.final_up(d1)       # (B,32,  H,   W)
        return self.final_conv(out)   # (B,1,   H,   W)  logits


# ── 3.7  Channel adapter: spatial encoder → FEAT_DIM ─────────────────────────

class ChannelAdapter(nn.Module):
    """
    Reduces the high-dimensional spatial encoder output (2048ch) to FEAT_DIM
    for use in prototype construction and the CDAM module.
    Used only for prototype/similarity computation — skip connections retain
    their original channels for the decoder.
    """
    def __init__(self, in_ch: int = 2048, feat_dim: int = 256):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv2d(in_ch, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.adapt(x)


# ── 3.8  FreqFSS — Full model ─────────────────────────────────────────────────

class FreqFSS(nn.Module):
    """
    FreqFSS: Frequency-Guided Prototype Alignment Few-Shot Segmentation.

    Forward pass API:
        model(support_imgs, support_masks, query_img)
        → predicted_mask logits (B, 1, H, W)
           + P_fused (B, C)  [for alignment loss]
           + P_f_prime (B, C) [for freq consistency loss]
           + Q_f (B, C, H, W) [for freq consistency loss]
           + query_mask_pred_probs [for freq consistency loss target region]

    All support and query images are encoded by the SAME encoder (shared weights).
    """
    def __init__(self, config: Config):
        super().__init__()
        C = config.FEAT_DIM

        self.spatial_enc = SpatialEncoder(pretrained=config.PRETRAINED)
        self.spectral_mlp = SpectralMLP(in_channels=6, feat_dim=C)
        self.channel_adapt = ChannelAdapter(in_ch=2048, feat_dim=C)
        self.abw   = AdaptiveBandWeighting(feat_dim=C, img_size=config.IMG_SIZE)
        self.cdam  = CrossDomainAlignmentModule(feat_dim=C, n_heads=config.ATTN_HEADS)
        self.decoder = FreqFSSDecoder()

    def encode_image(self, x):
        """
        Encodes a single image through both pathways.
        Returns:
            (s0, s1, s2, s3, s4)  spatial skip features
            F_s_adapted            (B, C, H/32, W/32)  for prototype
            F_f                    (B, C, H, W)         frequency features at full res
        """
        s0, s1, s2, s3, s4 = self.spatial_enc(x)
        F_s = self.channel_adapt(s4)          # (B, C, H/32, W/32)

        # Frequency encoder operates on original resolution image
        F_f = self.spectral_mlp(x)            # (B, C, H, W)

        return (s0, s1, s2, s3, s4), F_s, F_f

    def forward(self, support_imgs, support_masks, query_img):
        """
        support_imgs  : (B, K, 3, H, W)
        support_masks : (B, K, 1, H, W)
        query_img     : (B, 3, H, W)

        Returns dict with keys:
            'logits'     : (B, 1, H, W)
            'P_fused'    : (B, C)
            'P_f_prime'  : (B, C)
            'Q_f'        : (B, C, H, W)
            'Q_s'        : (B, C, H, W)   (adapted, for alignment loss)
        """
        B, K, _, H, W = support_imgs.shape

        # ── Encode K support images (batch-wise for efficiency) ──────────────
        sup_F_s_list, sup_F_f_list, sup_mask_list = [], [], []

        for k in range(K):
            sup_k   = support_imgs[:, k]         # (B, 3, H, W)
            mask_k  = support_masks[:, k]        # (B, 1, H, W)
            _, F_s_k, F_f_k = self.encode_image(sup_k)
            # Resize frequency features to match spatial feature map resolution
            H_s, W_s = F_s_k.shape[2], F_s_k.shape[3]
            F_f_k_down = F.interpolate(F_f_k, size=(H_s, W_s),
                                        mode='bilinear', align_corners=False)
            sup_F_s_list.append(F_s_k)
            sup_F_f_list.append(F_f_k_down)
            sup_mask_list.append(mask_k)

        # ── Build dual prototypes (averaged over K shots) ────────────────────
        P_s, P_f_raw = build_prototypes(sup_F_s_list, sup_F_f_list, sup_mask_list)
        # shape: (B, C)

        # ── Encode query image ───────────────────────────────────────────────
        (q_s0, q_s1, q_s2, q_s3, q_s4), Q_s, Q_f_full = self.encode_image(query_img)

        # Resize Q_f to match Q_s spatial resolution for CDAM
        H_q, W_q = Q_s.shape[2], Q_s.shape[3]
        Q_f = F.interpolate(Q_f_full, size=(H_q, W_q),
                             mode='bilinear', align_corners=False)

        # ── Adaptive Band Weighting on frequency prototype features ──────────
        # We apply ABW to the query frequency feature map conditioned on P_s
        Q_f_weighted = self.abw(Q_f, P_s)           # (B, C, H_q, W_q)

        # Also weight the frequency prototype using its own band decomposition
        # by globally pooling the weighted map
        P_f = (Q_f_weighted * F.adaptive_avg_pool2d(
            Q_f_weighted, 1)).sum(dim=(2,3)).unsqueeze(-1).unsqueeze(-1)
        P_f = P_f.squeeze(-1).squeeze(-1)
        # Fallback: use raw frequency prototype if weighting collapses
        # (safe blending: 0.7 weighted + 0.3 raw)
        P_f = 0.7 * P_f_raw + 0.3 * F.normalize(P_f_raw, dim=1)

        # ── Cross-Domain Alignment Module ────────────────────────────────────
        S, P_fused, P_f_prime = self.cdam(P_s, P_f, Q_s, Q_f_weighted)
        # S : (B, 1, H_q, W_q)

        # ── Prototype-guided decoder ─────────────────────────────────────────
        logits = self.decoder(q_s0, q_s1, q_s2, q_s3, q_s4, S)

        return {
            'logits':    logits,
            'P_fused':   P_fused,
            'P_f_prime': P_f_prime,
            'Q_f':       Q_f_weighted,
            'Q_s':       Q_s,
            'P_s':       P_s,
            'S':         S,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(pred_prob: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """Standard soft Dice loss for binary segmentation."""
    pred_flat   = pred_prob.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / \
                 (pred_flat.sum() + target_flat.sum() + smooth)


def seg_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Combined BCE + Dice loss (standard medical segmentation loss)."""
    probs = torch.sigmoid(logits)
    bce   = F.binary_cross_entropy_with_logits(logits, masks)
    dice  = dice_loss(probs, masks)
    return bce + dice


def freq_consistency_loss(P_f_prime: torch.Tensor,
                           Q_f: torch.Tensor,
                           pred_mask: torch.Tensor) -> torch.Tensor:
    """
    Spectral consistency loss:
    The predicted foreground region in Q_f should have similar spectral
    characteristics to P_f_prime (the frequency prototype).

    L_freq = || P_f' - MaskedPool(Q_f, M_pred) ||_2

    Research motivation: if the model correctly identifies the lesion, the
    spectral fingerprint of that region should match the prototype's spectral
    fingerprint. This acts as a self-supervised signal from the frequency branch.
    """
    B, C, H, W = Q_f.shape
    M = F.interpolate(pred_mask, size=(H, W), mode='bilinear',
                       align_corners=False)
    M = (M > 0.5).float()

    denom = M.sum(dim=(2, 3)).clamp(min=1.0)
    Q_f_masked_pool = (Q_f * M).sum(dim=(2, 3)) / denom   # (B, C)

    return F.mse_loss(Q_f_masked_pool, P_f_prime.detach())


def prototype_alignment_loss(P_s: torch.Tensor,
                              Q_s: torch.Tensor,
                              pred_mask: torch.Tensor) -> torch.Tensor:
    """
    Prototype alignment loss:
    The predicted foreground region in Q_s should be close to P_s in feature
    space (cosine distance).

    L_align = 1 - cosine(P_s, MaskedPool(Q_s, M_pred))

    This prevents prototype drift: the model is penalized if the region it
    identifies as foreground is dissimilar to the support prototype.
    """
    B, C, H, W = Q_s.shape
    M = F.interpolate(pred_mask, size=(H, W), mode='bilinear',
                       align_corners=False)
    M = (M > 0.5).float()

    denom = M.sum(dim=(2, 3)).clamp(min=1.0)
    Q_s_pool = (Q_s * M).sum(dim=(2, 3)) / denom   # (B, C)

    cos_sim = F.cosine_similarity(P_s, Q_s_pool, dim=1)    # (B,)
    return (1.0 - cos_sim).mean()


def total_loss(outputs: dict, query_mask: torch.Tensor,
               lambda_freq: float = 0.3,
               lambda_align: float = 0.2) -> dict:
    """
    Computes the full FreqFSS loss:
        L = L_seg + λ_freq * L_freq + λ_align * L_align
    """
    logits = outputs['logits']
    probs  = torch.sigmoid(logits)

    L_seg   = seg_loss(logits, query_mask)
    L_freq  = freq_consistency_loss(
        outputs['P_f_prime'], outputs['Q_f'], probs.detach()
    )
    L_align = prototype_alignment_loss(
        outputs['P_s'], outputs['Q_s'], probs.detach()
    )

    L_total = L_seg + lambda_freq * L_freq + lambda_align * L_align

    return {
        'total': L_total,
        'seg':   L_seg,
        'freq':  L_freq,
        'align': L_align,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — METRICS
# ══════════════════════════════════════════════════════════════════════════════


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff Distance for a single 2D image."""
    if np.sum(pred) == 0 and np.sum(target) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return 256.0 # Max typical distance
        
    pred_edges = pred ^ binary_erosion(pred, structure=np.ones((3,3)))
    target_edges = target ^ binary_erosion(target, structure=np.ones((3,3)))
    
    pred_pts = np.argwhere(pred_edges)
    target_pts = np.argwhere(target_edges)
    
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 256.0
        
    tree_pred = cKDTree(pred_pts)
    tree_target = cKDTree(target_pts)
    
    dist_pred_to_target, _ = tree_target.query(pred_pts)
    dist_target_to_pred, _ = tree_pred.query(target_pts)
    
    if len(dist_pred_to_target) == 0 or len(dist_target_to_pred) == 0:
        return 256.0
        
    hd95_val = max(np.percentile(dist_pred_to_target, 95), 
                   np.percentile(dist_target_to_pred, 95))
    return float(hd95_val)

def compute_metrics(pred_probs: np.ndarray, targets: np.ndarray) -> dict:

    """
    Binary segmentation metrics.
    Both inputs: (N, H, W) arrays with values in [0,1].
    pred_probs thresholded at 0.5.
    """
    preds = (pred_probs > 0.5).astype(np.uint8)
    tgts  = targets.astype(np.uint8)

    p = preds.flatten()
    t = tgts.flatten()

    tp = int(((p == 1) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou  = tp / (tp + fp + fn + 1e-7)
    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)

    hd95_list = [compute_hd95(preds[i], tgts[i]) for i in range(preds.shape[0])]
    hd95 = float(np.mean(hd95_list)) if hd95_list else 0.0

    return dict(dice=float(dice), iou=float(iou),
                sensitivity=float(sens), specificity=float(spec), hd95=float(hd95))


@torch.no_grad()
def evaluate_episodes(model: nn.Module, loader: DataLoader,
                       device: torch.device) -> dict:
    """
    Runs evaluation over all episodes in loader and returns mean metrics.
    """
    model.eval()
    all_probs, all_targets = [], []

    for support_imgs, support_masks, query_img, query_mask in tqdm(
            loader, desc='  Evaluating', leave=False, ncols=90):

        support_imgs  = support_imgs.to(device)
        support_masks = support_masks.to(device)
        query_img     = query_img.to(device)

        outputs = model(support_imgs, support_masks, query_img)
        probs   = torch.sigmoid(outputs['logits']).cpu().numpy()  # (B,1,H,W)
        targets = query_mask.numpy()                               # (B,1,H,W)

        all_probs.append(probs.squeeze(1))      # (B,H,W)
        all_targets.append(targets.squeeze(1))  # (B,H,W)

    all_probs   = np.concatenate(all_probs,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return compute_metrics(all_probs, all_targets)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def cosine_lr(optimizer, epoch: int, max_epochs: int,
              base_lr: float = 1e-4, min_lr: float = 1e-6):
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / max_epochs))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    config: Config,
                    epoch: int) -> dict:
    """Single training epoch over all episodes."""
    model.train()
    epoch_losses = {'total': [], 'seg': [], 'freq': [], 'align': []}

    pbar = tqdm(loader, desc=f'  [Train] Ep {epoch:3d}/{config.EPOCHS}',
                leave=False, ncols=110)

    for support_imgs, support_masks, query_img, query_mask in pbar:
        support_imgs  = support_imgs.to(config.DEVICE)
        support_masks = support_masks.to(config.DEVICE)
        query_img     = query_img.to(config.DEVICE)
        query_mask    = query_mask.to(config.DEVICE)

        optimizer.zero_grad()

        outputs = model(support_imgs, support_masks, query_img)
        losses  = total_loss(outputs, query_mask,
                             config.LAMBDA_FREQ, config.LAMBDA_ALIGN)

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k in epoch_losses:
            epoch_losses[k].append(losses[k].item())

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'seg':  f"{losses['seg'].item():.4f}",
            'freq': f"{losses['freq'].item():.4f}",
        })

    return {k: float(np.mean(v)) for k, v in epoch_losses.items()}


def train_freqfss(config: Config = None, epochs: int = None) -> dict:
    """
    Main training function. Call directly from a Kaggle notebook cell.

    Args:
        config : Config object (default: Config() with all defaults)
        epochs : override Config.EPOCHS (useful for quick sanity checks)

    Returns:
        dict with keys 'val_metrics', 'test_metrics', 'history'
    """
    if config is None:
        config = Config()
    if epochs is not None:
        config.EPOCHS = epochs

    set_seed(config.SEED)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  FreqFSS — Few-Shot Segmentation on ISIC 2016")
    print(f"  Device : {config.DEVICE}")
    print(f"  Epochs : {config.EPOCHS}  |  K-shot : {config.K_SHOT}")
    print(f"{'═'*65}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data ...")
    train_loader, val_loader, test_loader = make_episode_loaders(config)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model ...")
    model = FreqFSS(config).to(config.DEVICE)

    # Multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate LR groups: pretrained backbone gets 10× lower LR (standard practice)
    backbone_params = []
    new_params      = []
    base_model = model.module if hasattr(model, 'module') else model
    for name, param in base_model.named_parameters():
        if 'spatial_enc' in name:
            backbone_params.append(param)
        else:
            new_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.LR / 10},
        {'params': new_params,      'lr': config.LR},
    ], weight_decay=config.WEIGHT_DECAY)

    # ── Training history ──────────────────────────────────────────────────────
    history = {
        'train_loss': [], 'train_seg': [], 'train_freq': [], 'train_align': [],
        'val_dice':   [], 'val_iou':   [],
        'val_sensitivity': [], 'val_specificity': [],
        'lr': [],
    }

    best_val_dice   = 0.0
    best_model_path = os.path.join(config.SAVE_DIR, 'freqfss_best.pth')

    # ── Epoch loop ────────────────────────────────────────────────────────────
    print(f"\nStarting training for {config.EPOCHS} epochs ...\n")
    for epoch in range(1, config.EPOCHS + 1):
        lr = cosine_lr(optimizer, epoch, config.EPOCHS,
                       base_lr=config.LR, min_lr=config.LR_MIN)
        history['lr'].append(lr)

        # Train
        train_losses = train_one_epoch(model, train_loader, optimizer, config, epoch)
        history['train_loss'].append(train_losses['total'])
        history['train_seg'].append(train_losses['seg'])
        history['train_freq'].append(train_losses['freq'])
        history['train_align'].append(train_losses['align'])

        # Validate every 5 epochs (and final epoch) to save time
        if epoch % 5 == 0 or epoch == config.EPOCHS:
            val_metrics = evaluate_episodes(model, val_loader, config.DEVICE)
            history['val_dice'].append(val_metrics['dice'])
            history['val_iou'].append(val_metrics['iou'])
            history['val_sensitivity'].append(val_metrics['sensitivity'])
            history['val_specificity'].append(val_metrics['specificity'])

            print(
                f"Epoch {epoch:3d}/{config.EPOCHS} | "
                f"LR {lr:.2e} | "
                f"Loss {train_losses['total']:.4f} "
                f"(seg {train_losses['seg']:.4f} "
                f"freq {train_losses['freq']:.4f} "
                f"align {train_losses['align']:.4f}) | "
                f"Val Dice {val_metrics['dice']:.4f} "
                f"IoU {val_metrics['iou']:.4f} "
                f"Sens {val_metrics['sensitivity']:.4f} "
                f"Spec {val_metrics['specificity']:.4f} "
                f"HD95 {val_metrics['hd95']:.4f}"
            )

            # Save best checkpoint
            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                save_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch':      epoch,
                    'state_dict': save_model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config':     config.__dict__,
                }, best_model_path)
                print(f"  ✓ Best model saved (Val Dice {best_val_dice:.4f})")
        else:
            print(
                f"Epoch {epoch:3d}/{config.EPOCHS} | "
                f"LR {lr:.2e} | "
                f"Loss {train_losses['total']:.4f} "
                f"(seg {train_losses['seg']:.4f} "
                f"freq {train_losses['freq']:.4f} "
                f"align {train_losses['align']:.4f})"
            )

    # ── Final test evaluation on best checkpoint ───────────────────────────────
    print(f"\n{'─'*65}")
    print("Loading best model for final test evaluation ...")
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=config.DEVICE)
        base_model = model.module if hasattr(model, 'module') else model
        base_model.load_state_dict(ckpt['state_dict'])
        print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
              f"(Val Dice {ckpt['val_metrics']['dice']:.4f})")

    print(f"\nRunning test evaluation over {config.EPISODES_TEST} episodes ...")
    test_metrics = evaluate_episodes(model, test_loader, config.DEVICE)

    print(f"\n{'═'*65}")
    print(f"  FINAL TEST RESULTS ({config.EPISODES_TEST} episodes, 5-shot)")
    print(f"  Dice        : {test_metrics['dice']:.4f}")
    print(f"  IoU         : {test_metrics['iou']:.4f}")
    print(f"  Sensitivity : {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity : {test_metrics['specificity']:.4f}")
    print(f"  HD95        : {test_metrics['hd95']:.4f}")
    print(f"{'═'*65}\n")

    # Save results
    results = {
        'test_metrics':  test_metrics,
        'best_val_dice': best_val_dice,
        'history':       history,
        'config': {
            'epochs': config.EPOCHS,
            'k_shot': config.K_SHOT,
            'lr':     config.LR,
            'lambda_freq':  config.LAMBDA_FREQ,
            'lambda_align': config.LAMBDA_ALIGN,
        },
    }
    results_path = os.path.join(config.SAVE_DIR, 'freqfss_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Usage options in a Kaggle notebook:

    Option A — Run everything with defaults (recommended):
        results = train_freqfss()

    Option B — Quick sanity check (10 epochs):
        results = train_freqfss(epochs=10)

    Option C — Custom config:
        cfg = Config()
        cfg.EPOCHS = 150
        cfg.K_SHOT = 5
        cfg.LAMBDA_FREQ = 0.4
        results = train_freqfss(config=cfg)

    Option D — Just evaluate a saved checkpoint:
        config = Config()
        model  = FreqFSS(config).to(config.DEVICE)
        ckpt   = torch.load('./freqfss_results/freqfss_best.pth')
        model.load_state_dict(ckpt['state_dict'])
        _, _, test_loader = make_episode_loaders(config)
        metrics = evaluate_episodes(model, test_loader, config.DEVICE)
        print(metrics)
    """
    results = train_freqfss()