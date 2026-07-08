# -*- coding: utf-8 -*-
"""
freqfss_busi.py
════════════════════════════════════════════════════════════════════════════════
FreqFSS: Frequency-Guided Prototype Alignment for Few-Shot MRI/US Segmentation
         — adapted for Breast Ultrasound Images Dataset (BUSI)

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

  Cross-Domain Alignment Module (CDAM):
    • Bidirectional cross-attention: P_s ↔ P_f → P_fused
    • Dual similarity maps: S_spatial (P_fused · Q_s) + S_freq (P_f · Q_f)
    • Learnable α,β fusion

DATA LOADING (Adapted for BUSI)
───────────────────────────────
  Dataset  : Breast Ultrasound Images Dataset (BUSI)
  Structure: Dataset_BUSI_with_GT/{benign, malignant, normal}/
  Mask name: {base}_mask.png
  Reading  : cv2 grayscale/RGB handling, binary thresholding
  Norm     : ImageNet mean/std, 256×256
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
from glob import glob

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
    # ── Paths (Updated for BUSI structure) ────────────────────────────────────
    DATA_PATH         = os.environ.get(
        'FREQFSS_DATA',
        '/kaggle/input/datasets/sabahesaraki/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT'
    )
    SAVE_DIR          = './freqfss_results_busi'

    # ── Image ─────────────────────────────────────────────────────────────────
    IMG_SIZE    = 256
    NUM_CLASSES = 1       # binary: tumor vs background

    # ── Episode protocol ──────────────────────────────────────────────────────
    N_WAY    = 1          # single-class binary segmentation
    K_SHOT   = 5          # 5 support images per episode
    N_QUERY  = 1          # queries per episode during training

    # ── Split ratios (Applied dynamically across the whole dataset) ───────────
    VAL_RATIO  = 0.15     # of train split
    TEST_RATIO = 0.20     # held-out test images

    # ── Training ──────────────────────────────────────────────────────────────
    EPOCHS          = 100
    EPISODES_TRAIN  = 200   
    EPISODES_VAL    = 200   
    EPISODES_TEST   = 1000  
    LR              = 1e-4
    LR_MIN          = 1e-6
    WEIGHT_DECAY    = 1e-4
    BATCH_SIZE      = 1     # 1 episode per forward pass
    NUM_WORKERS     = 0
    PIN_MEMORY      = False

    # ── Loss weights ──────────────────────────────────────────────────────────
    LAMBDA_FREQ  = 0.3    
    LAMBDA_ALIGN = 0.2    

    # ── Architecture ──────────────────────────────────────────────────────────
    FEAT_DIM      = 256   
    FREQ_BANDS    = 3     
    ATTN_HEADS    = 4     
    PRETRAINED    = True  

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# SECTION 2 — DATA LOADING (BUSI SPECIFIC)
# ══════════════════════════════════════════════════════════════════════════════

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

def support_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5), # Slightly less rotation for US scans
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.5),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])

def query_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.9, 1.0), p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0, p=0.6),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])

def val_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def build_busi_index(config: Config):
    """
    Scans the BUSI dataset directories: benign, malignant, normal.
    Matches e.g., 'benign (1).png' with 'benign (1)_mask.png'.
    Note: BUSI sometimes contains multiple masks (mask_1, mask_2). 
    This logic safely loads the primary '_mask.png' to ensure consistency.
    """
    img_paths, mask_paths = [], []
    classes = ['benign', 'malignant', 'normal']
    
    for cls in classes:
        cls_dir = os.path.join(config.DATA_PATH, cls)
        if not os.path.exists(cls_dir):
            print(f"  ⚠ Directory not found: {cls_dir}")
            continue
            
        # Find all images that are NOT masks
        for img_path in sorted(glob(os.path.join(cls_dir, '*.png'))):
            if '_mask' in img_path:
                continue
                
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(cls_dir, f"{base_name}_mask.png")
            
            if os.path.exists(mask_path):
                img_paths.append(img_path)
                mask_paths.append(mask_path)

    print(f"  ✓ Total paired samples found: {len(img_paths)}")
    return img_paths, mask_paths


def split_dataset(img_paths, mask_paths, config: Config, seed: int = 42):
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
        return [img_paths[i] for i in idx_list], [mask_paths[i] for i in idx_list]

    print(f"  Train: {len(train_idx):>5}  |  Val: {len(val_idx):>4}  |  Test: {len(test_idx):>4}")
    return subset(train_idx), subset(val_idx), subset(test_idx)


def load_image_mask(img_path: str, mask_path: str):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    mask = (mask > 127).astype(np.float32)
    return image, mask


class EpisodicBUSIDataset(Dataset):
    def __init__(
        self, img_paths, mask_paths,
        k_shot: int = 5, n_episodes: int = 1000,
        img_size: int = 256, mode: str = 'train', seed: int = 42,
    ):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.k_shot     = k_shot
        self.n_episodes = n_episodes
        
        rng = random.Random(seed)
        self.episodes = []
        n = len(img_paths)
        for _ in range(n_episodes):
            indices = rng.sample(range(n), k_shot + 1)
            self.episodes.append((indices[:k_shot], indices[k_shot]))

        if mode == 'train':
            self.sup_tf, self.query_tf = support_transform(img_size), query_transform(img_size)
        else:
            self.sup_tf = self.query_tf = val_transform(img_size)

    def __len__(self): return self.n_episodes

    def _load(self, img_p, mask_p, tf):
        img, mask = load_image_mask(img_p, mask_p)
        aug = tf(image=img, mask=mask)
        return aug['image'], aug['mask'].unsqueeze(0).float()

    def __getitem__(self, idx):
        sup_idx, q_idx = self.episodes[idx]
        sup_imgs, sup_masks = zip(*[self._load(self.img_paths[i], self.mask_paths[i], self.sup_tf) for i in sup_idx])
        q_img, q_mask = self._load(self.img_paths[q_idx], self.mask_paths[q_idx], self.query_tf)
        return torch.stack(sup_imgs), torch.stack(sup_masks), q_img, q_mask


def make_episode_loaders(config: Config):
    img_paths, mask_paths = build_busi_index(config)
    (tr_i, tr_m), (va_i, va_m), (te_i, te_m) = split_dataset(img_paths, mask_paths, config, seed=config.SEED)

    loader_kw = dict(num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    
    train_ds = EpisodicBUSIDataset(tr_i, tr_m, config.K_SHOT, config.EPISODES_TRAIN, config.IMG_SIZE, 'train', config.SEED)
    val_ds   = EpisodicBUSIDataset(va_i, va_m, config.K_SHOT, config.EPISODES_VAL, config.IMG_SIZE, 'val', config.SEED + 1)
    test_ds  = EpisodicBUSIDataset(te_i, te_m, config.K_SHOT, config.EPISODES_TEST, config.IMG_SIZE, 'test', config.SEED + 2)

    return (
        DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, **loader_kw),
        DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kw),
        DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kw)
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ARCHITECTURE (Unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        return s0, s1, s2, s3, s4

class SpectralMLP(nn.Module):
    def __init__(self, in_channels: int = 6, feat_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_dim, kernel_size=1, bias=False), nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_fft = torch.fft.fft2(x, norm='ortho')
        magnitude = torch.log1p(torch.abs(x_fft))
        phase = torch.angle(x_fft)
        return self.mlp(torch.cat([magnitude, phase], dim=1))

class AdaptiveBandWeighting(nn.Module):
    def __init__(self, feat_dim: int = 256, img_size: int = 256):
        super().__init__()
        self.band_mlp = nn.Sequential(nn.Linear(feat_dim * 4, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))
        self.register_buffer('ring_masks', self._build_ring_masks(img_size))

    @staticmethod
    def _build_ring_masks(img_size: int) -> torch.Tensor:
        H, W = img_size, img_size
        r = torch.sqrt((torch.arange(H).float().unsqueeze(1).expand(H, W) - H/2.0)**2 + 
                       (torch.arange(W).float().unsqueeze(0).expand(H, W) - W/2.0)**2)
        r_max = min(H, W) / 2.0
        masks = torch.zeros(3, H, W)
        masks[0] = (r < r_max / 3).float()
        masks[1] = ((r >= r_max / 3) & (r < 2 * r_max / 3)).float()
        masks[2] = (r >= 2 * r_max / 3).float()
        return masks

    def forward(self, F_f: torch.Tensor, P_s: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F_f.shape
        ring_masks = F.interpolate(self.ring_masks.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
        
        ring_descs = [(F_f * ring_masks[k].view(1,1,H,W)).sum(dim=(2, 3)) / ring_masks[k].sum().clamp(min=1.0) for k in range(3)]
        weights = torch.softmax(self.band_mlp(torch.cat([P_s] + ring_descs, dim=1)), dim=1)

        out = torch.zeros_like(F_f)
        for k in range(3):
            out += weights[:, k].view(B, 1, 1, 1) * F_f * ring_masks[k].view(1,1,H,W)
        return out

class CrossDomainAlignmentModule(nn.Module):
    def __init__(self, feat_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.q_proj_s, self.k_proj_s, self.v_proj_s = nn.Linear(feat_dim, feat_dim), nn.Linear(feat_dim, feat_dim), nn.Linear(feat_dim, feat_dim)
        self.q_proj_f, self.k_proj_f, self.v_proj_f = nn.Linear(feat_dim, feat_dim), nn.Linear(feat_dim, feat_dim), nn.Linear(feat_dim, feat_dim)
        self.n_heads, self.head_dim = n_heads, feat_dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.norm_s, self.norm_f = nn.LayerNorm(feat_dim), nn.LayerNorm(feat_dim)
        self.alpha, self.beta = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))

    def _cross_attn(self, q_vec, k_vec, v_vec, q_proj, k_proj, v_proj):
        B, H, Hd = q_vec.size(0), self.n_heads, self.head_dim
        Q, K, V = q_proj(q_vec).view(B, H, Hd), k_proj(k_vec).view(B, H, Hd), v_proj(v_vec).view(B, H, Hd)
        attn = torch.softmax((Q * K).sum(dim=-1, keepdim=True) / self.scale, dim=1)
        return (attn * V).view(B, -1)

    def forward(self, P_s, P_f, Q_s, Q_f):
        B, C, H, W = Q_s.shape
        P_s_prime = self.norm_s(P_s + self._cross_attn(P_s, P_f, P_f, self.q_proj_s, self.k_proj_s, self.v_proj_s))
        P_f_prime = self.norm_f(P_f + self._cross_attn(P_f, P_s, P_s, self.q_proj_f, self.k_proj_f, self.v_proj_f))
        
        P_fused = P_s_prime + P_f_prime
        
        S_spatial = (F.normalize(P_fused.unsqueeze(-1), dim=1) * F.normalize(Q_s.view(B, C, -1), dim=1)).sum(dim=1).view(B, 1, H, W)
        S_freq = (F.normalize(P_f_prime.unsqueeze(-1), dim=1) * F.normalize(Q_f.view(B, C, -1), dim=1)).sum(dim=1).view(B, 1, H, W)
        
        return torch.sigmoid(self.alpha) * S_spatial + torch.sigmoid(self.beta) * S_freq, P_fused, P_f_prime

def build_prototypes(F_s_list, F_f_list, mask_list):
    ps_list, pf_list = [], []
    for F_s, F_f, M in zip(F_s_list, F_f_list, mask_list):
        M_down = (F.interpolate(M.float(), size=F_s.shape[2:], mode='bilinear', align_corners=False) > 0.5).float()
        denom = M_down.sum(dim=(2, 3)).clamp(min=1.0)
        ps_list.append((F_s * M_down).sum(dim=(2, 3)) / denom)
        pf_list.append((F_f * M_down).sum(dim=(2, 3)) / denom)
    return torch.stack(ps_list, dim=0).mean(dim=0), torch.stack(pf_list, dim=0).mean(dim=0)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(ConvBNReLU(in_ch, out_ch), ConvBNReLU(out_ch, out_ch))
    def forward(self, x): return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape: x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))

class BottleneckBlock(nn.Module):
    def __init__(self, in_ch: int = 2049, out_ch: int = 1024):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, s4, S):
        S_resized = F.interpolate(S, size=s4.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([s4, S_resized], dim=1))

class FreqFSSDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = BottleneckBlock(in_ch=2049, out_ch=1024)
        self.up4 = UpBlock(1024, 1024, 512)
        self.up3 = UpBlock(512,  512,  256)
        self.up2 = UpBlock(256,  256,  128)
        self.up1 = UpBlock(128,  64,   64)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
    def forward(self, s0, s1, s2, s3, s4, S):
        b = self.bottleneck(s4, S)
        return self.final_conv(self.final_up(self.up1(self.up2(self.up3(self.up4(b, s3), s2), s1), s0)))

class ChannelAdapter(nn.Module):
    def __init__(self, in_ch: int = 2048, feat_dim: int = 256):
        super().__init__()
        self.adapt = nn.Sequential(nn.Conv2d(in_ch, feat_dim, kernel_size=1, bias=False), nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True))
    def forward(self, x): return self.adapt(x)

class FreqFSS(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.spatial_enc = SpatialEncoder(pretrained=config.PRETRAINED)
        self.spectral_mlp = SpectralMLP(in_channels=6, feat_dim=config.FEAT_DIM)
        self.channel_adapt = ChannelAdapter(in_ch=2048, feat_dim=config.FEAT_DIM)
        self.abw = AdaptiveBandWeighting(feat_dim=config.FEAT_DIM, img_size=config.IMG_SIZE)
        self.cdam = CrossDomainAlignmentModule(feat_dim=config.FEAT_DIM, n_heads=config.ATTN_HEADS)
        self.decoder = FreqFSSDecoder()

    def encode_image(self, x):
        s = self.spatial_enc(x)
        return s, self.channel_adapt(s[-1]), self.spectral_mlp(x)

    def forward(self, support_imgs, support_masks, query_img):
        B, K, _, _, _ = support_imgs.shape
        sup_F_s_list, sup_F_f_list, sup_mask_list = [], [], []

        for k in range(K):
            s_k, F_s_k, F_f_k = self.encode_image(support_imgs[:, k])
            sup_F_s_list.append(F_s_k)
            sup_F_f_list.append(F.interpolate(F_f_k, size=F_s_k.shape[2:], mode='bilinear', align_corners=False))
            sup_mask_list.append(support_masks[:, k])

        P_s, P_f_raw = build_prototypes(sup_F_s_list, sup_F_f_list, sup_mask_list)
        q_s, Q_s, Q_f_full = self.encode_image(query_img)
        Q_f = F.interpolate(Q_f_full, size=Q_s.shape[2:], mode='bilinear', align_corners=False)

        Q_f_weighted = self.abw(Q_f, P_s)
        P_f = (Q_f_weighted * F.adaptive_avg_pool2d(Q_f_weighted, 1)).sum(dim=(2,3)).unsqueeze(-1).unsqueeze(-1).squeeze(-1).squeeze(-1)
        P_f = 0.7 * P_f + 0.3 * F.normalize(P_f_raw, dim=1)

        S, P_fused, P_f_prime = self.cdam(P_s, P_f, Q_s, Q_f_weighted)
        return {
            'logits': self.decoder(*q_s, S),
            'P_fused': P_fused, 'P_f_prime': P_f_prime, 'Q_f': Q_f_weighted, 'Q_s': Q_s, 'P_s': P_s, 'S': S,
        }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 & 5 & 6 & 7 — LOSSES, METRICS, TRAINING (Unchanged Logic)
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(pred_prob, target, smooth=1.0):
    p, t = pred_prob.reshape(-1), target.reshape(-1)
    return 1.0 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)

def seg_loss(logits, masks):
    return F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(torch.sigmoid(logits), masks)

def freq_consistency_loss(P_f_prime, Q_f, pred_mask):
    M = (F.interpolate(pred_mask, size=Q_f.shape[2:], mode='bilinear', align_corners=False) > 0.5).float()
    return F.mse_loss((Q_f * M).sum(dim=(2, 3)) / M.sum(dim=(2, 3)).clamp(min=1.0), P_f_prime.detach())

def prototype_alignment_loss(P_s, Q_s, pred_mask):
    M = (F.interpolate(pred_mask, size=Q_s.shape[2:], mode='bilinear', align_corners=False) > 0.5).float()
    return (1.0 - F.cosine_similarity(P_s, (Q_s * M).sum(dim=(2, 3)) / M.sum(dim=(2, 3)).clamp(min=1.0), dim=1)).mean()

def total_loss(outputs, query_mask, lambda_freq=0.3, lambda_align=0.2):
    probs = torch.sigmoid(outputs['logits'])
    l_seg = seg_loss(outputs['logits'], query_mask)
    l_freq = freq_consistency_loss(outputs['P_f_prime'], outputs['Q_f'], probs.detach())
    l_align = prototype_alignment_loss(outputs['P_s'], outputs['Q_s'], probs.detach())
    return {'total': l_seg + lambda_freq * l_freq + lambda_align * l_align, 'seg': l_seg, 'freq': l_freq, 'align': l_align}

def compute_metrics(pred_probs, targets):
    p, t = (pred_probs > 0.5).astype(np.uint8).flatten(), targets.astype(np.uint8).flatten()
    tp, tn, fp, fn = ((p==1)&(t==1)).sum(), ((p==0)&(t==0)).sum(), ((p==1)&(t==0)).sum(), ((p==0)&(t==1)).sum()
    return {
        'dice': float((2*tp) / (2*tp + fp + fn + 1e-7)),
        'iou': float(tp / (tp + fp + fn + 1e-7)),
        'sensitivity': float(tp / (tp + fn + 1e-7)),
        'specificity': float(tn / (tn + fp + 1e-7))
    }

@torch.no_grad()
def evaluate_episodes(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    for s_i, s_m, q_i, q_m in tqdm(loader, desc='  Evaluating', leave=False, ncols=90):
        outputs = model(s_i.to(device), s_m.to(device), q_i.to(device))
        all_probs.append(torch.sigmoid(outputs['logits']).cpu().numpy().squeeze(1))
        all_targets.append(q_m.numpy().squeeze(1))
    return compute_metrics(np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0))

def cosine_lr(optimizer, epoch, max_epochs, base_lr=1e-4, min_lr=1e-6):
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / max_epochs))
    for pg in optimizer.param_groups: pg['lr'] = lr
    return lr

def train_one_epoch(model, loader, optimizer, config, epoch):
    model.train()
    epoch_losses = {'total': [], 'seg': [], 'freq': [], 'align': []}
    pbar = tqdm(loader, desc=f'  [Train] Ep {epoch:3d}/{config.EPOCHS}', leave=False, ncols=110)

    for s_i, s_m, q_i, q_m in pbar:
        optimizer.zero_grad()
        outputs = model(s_i.to(config.DEVICE), s_m.to(config.DEVICE), q_i.to(config.DEVICE))
        losses = total_loss(outputs, q_m.to(config.DEVICE), config.LAMBDA_FREQ, config.LAMBDA_ALIGN)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        for k in epoch_losses: epoch_losses[k].append(losses[k].item())
        pbar.set_postfix({'loss': f"{losses['total'].item():.4f}", 'seg': f"{losses['seg'].item():.4f}"})
    return {k: float(np.mean(v)) for k, v in epoch_losses.items()}

def train_freqfss(config: Config = None, epochs: int = None):
    if config is None: config = Config()
    if epochs is not None: config.EPOCHS = epochs
    set_seed(config.SEED)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print(f"\n{'═'*65}\n  FreqFSS — Few-Shot Segmentation on BUSI\n  Device : {config.DEVICE}\n{'═'*65}\n")
    train_loader, val_loader, test_loader = make_episode_loaders(config)
    model = nn.DataParallel(FreqFSS(config)).to(config.DEVICE) if torch.cuda.device_count() > 1 else FreqFSS(config).to(config.DEVICE)

    backbone_params, new_params = [], []
    for name, param in (model.module if hasattr(model, 'module') else model).named_parameters():
        (backbone_params if 'spatial_enc' in name else new_params).append(param)

    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': config.LR / 10}, {'params': new_params, 'lr': config.LR}], weight_decay=config.WEIGHT_DECAY)

    best_val_dice, best_model_path = 0.0, os.path.join(config.SAVE_DIR, 'freqfss_best_busi.pth')

    for epoch in range(1, config.EPOCHS + 1):
        lr = cosine_lr(optimizer, epoch, config.EPOCHS, base_lr=config.LR, min_lr=config.LR_MIN)
        train_losses = train_one_epoch(model, train_loader, optimizer, config, epoch)

        if epoch % 5 == 0 or epoch == config.EPOCHS:
            val_metrics = evaluate_episodes(model, val_loader, config.DEVICE)
            print(f"Ep {epoch:3d} | Loss {train_losses['total']:.4f} | Val Dice {val_metrics['dice']:.4f}")
            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                torch.save({'state_dict': (model.module if hasattr(model, 'module') else model).state_dict()}, best_model_path)
        else:
            print(f"Ep {epoch:3d} | Loss {train_losses['total']:.4f}")

    if os.path.exists(best_model_path):
        (model.module if hasattr(model, 'module') else model).load_state_dict(torch.load(best_model_path, map_location=config.DEVICE)['state_dict'])
    
    test_metrics = evaluate_episodes(model, test_loader, config.DEVICE)
    print(f"\n{'═'*65}\n  FINAL TEST RESULTS\n  Dice: {test_metrics['dice']:.4f}  |  IoU: {test_metrics['iou']:.4f}\n{'═'*65}\n")
    return test_metrics

if __name__ == '__main__':
    results = train_freqfss()