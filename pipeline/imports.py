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
from glob import glob

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
