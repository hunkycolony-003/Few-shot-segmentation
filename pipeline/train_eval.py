from imports import *
from model import FreqFSS

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

    return dict(dice=float(dice), iou=float(iou),
                sensitivity=float(sens), specificity=float(spec))


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

def cosine_lr(optimizer, epoch: int, max_epochs: int,
              base_lr: float = 1e-4, min_lr: float = 1e-6):
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / max_epochs))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    config: 'Config',
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


def train_freqfss(config, dataset_name: str, epochs: int = None, make_episode_loaders_fn=None) -> dict:
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
    print(f"  FreqFSS — Few-Shot Segmentation on {dataset_name}")
    print(f"  Device : {config.DEVICE}")
    print(f"  Epochs : {config.EPOCHS}  |  K-shot : {config.K_SHOT}")
    print(f"{'═'*65}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data ...")
    train_loader, val_loader, test_loader = make_episode_loaders_fn(config)

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

    if getattr(config, 'FREEZE_BACKBONE', False):
        print("  Freezing spatial encoder (backbone) parameters ...")
        for p in backbone_params:
            p.requires_grad = False
            
        optimizer = torch.optim.AdamW([
            {'params': new_params,      'lr': config.LR},
        ], weight_decay=config.WEIGHT_DECAY)
    else:
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
    best_model_path = os.path.join(config.SAVE_DIR, f'freqfss_best_{dataset_name}.pth')

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
                f"Spec {val_metrics['specificity']:.4f}"
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
