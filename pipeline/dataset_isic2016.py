from imports import *
from train_eval import train_freqfss

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
    SAVE_DIR          = './freqfss_results_isic2016'

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
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n[Local Sanity Check Mode] Overriding config for a fast local test...")
    config = Config()
    
    # ── Fast Sanity Check Overrides ──
    config.EPOCHS = 1
    config.EPISODES_TRAIN = 2
    config.EPISODES_VAL = 1
    config.EPISODES_TEST = 1
    config.FREEZE_BACKBONE = True
    config.DATA_PATH = os.environ.get(
        'FREQFSS_DATA',
        'Data/segmentation-isic-ham10k'
    )
    config.SEG_PATH = os.path.join(config.DATA_PATH, 'ISIC16_Resized256x256')
    config.TRAIN_IMG_DIR = os.path.join(config.SEG_PATH, 'Train', 'Images')
    config.TRAIN_MASK_DIR = os.path.join(config.SEG_PATH, 'Train', 'Masks')
    config.TEST_IMG_DIR = os.path.join(config.SEG_PATH, 'Test', 'Images')
    config.TEST_MASK_DIR = os.path.join(config.SEG_PATH, 'Test', 'Masks')
    
    # Ensure data path points to a local directory if you are testing locally. 
    # If the Kaggle path doesn't exist, you might need to mock it or pass a tiny sample dataset.
    
    train_freqfss(
        config=config,
        dataset_name=config.__class__.__name__.replace("Config", "").upper() if hasattr(config, "__class__") else "DATASET",
        make_episode_loaders_fn=make_episode_loaders
    )
