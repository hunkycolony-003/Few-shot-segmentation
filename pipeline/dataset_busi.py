from imports import *
from train_eval import train_freqfss

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
        'Data/Dataset_BUSI_with_GT'
    )
    
    # Ensure data path points to a local directory if you are testing locally. 
    # If the Kaggle path doesn't exist, you might need to mock it or pass a tiny sample dataset.
    
    train_freqfss(
        config=config,
        dataset_name=config.__class__.__name__.replace("Config", "").upper() if hasattr(config, "__class__") else "DATASET",
        make_episode_loaders_fn=make_episode_loaders
    )
