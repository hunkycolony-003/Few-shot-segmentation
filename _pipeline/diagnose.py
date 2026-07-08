import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import TumorDataset, EpisodicBatchSampler
from dual_branch_panet import DualBranchPANet, masked_pooling, compute_similarity_map
from train import get_device


def diagnose_model(model, device, tag=""):
    """Print learned parameter values."""
    print(f"\n{'='*60}")
    print(f"MODEL DIAGNOSTICS {tag}")
    print(f"{'='*60}")
    
    # 1. Fusion weights
    w1 = model.fusion_weight1.item()
    w2 = model.fusion_weight2.item()
    w_fused = 1 - w1 - w2
    print(f"\n--- Learned Fusion Weights ---")
    print(f"  fusion_weight1 (freq):    {w1:.4f}")
    print(f"  fusion_weight2 (spatial): {w2:.4f}")
    print(f"  fused weight (1-w1-w2):   {w_fused:.4f}")
    print(f"  scaler:                   {model.scaler.item():.4f}")
    
    # 2. Band weighting MLP stats
    print(f"\n--- Band Weighting MLP ---")
    for name, param in model.band_weighting.named_parameters():
        print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}, norm={param.data.norm().item():.4f}")
    
    # 3. Frequency encoder conv layer stats
    print(f"\n--- Frequency Encoder Conv Layers ---")
    print(f"  conv1 weight norm: {model.freq_encoder.conv1.weight.data.norm().item():.4f}")
    print(f"  conv2 weight norm: {model.freq_encoder.conv2.weight.data.norm().item():.4f}")
    

def diagnose_episode(model, support_img, query_img, support_mask, target_mask, device, fold, ep_idx):
    """Run a single episode with detailed intermediate logging."""
    model.eval()
    
    K = support_img.shape[1]
    B = support_img.shape[0]
    
    with torch.no_grad():
        # Flatten support
        s_img_flat = support_img.view(B * K, 3, support_img.shape[3], support_img.shape[4])
        fore_mask_flat = support_mask.view(B * K, 1, support_mask.shape[3], support_mask.shape[4])
        back_mask_flat = 1 - fore_mask_flat
        
        # Extract features
        f_s = model.spatial_encoder(s_img_flat)
        f_f = model.freq_encoder(s_img_flat)
        q_s = model.spatial_encoder(query_img)
        q_f = model.freq_encoder(query_img)
        
        print(f"\n--- Episode {ep_idx+1} (Fold {fold}) Feature Analysis ---")
        
        # Feature magnitudes
        print(f"  Support spatial features (f_s): mean={f_s.mean():.4f}, std={f_s.std():.4f}, norm={f_s.norm():.2f}")
        print(f"  Support freq features (f_f):    mean={f_f.mean():.4f}, std={f_f.std():.4f}, norm={f_f.norm():.2f}")
        print(f"  Query spatial features (q_s):   mean={q_s.mean():.4f}, std={q_s.std():.4f}, norm={q_s.norm():.2f}")
        print(f"  Query freq features (q_f):      mean={q_f.mean():.4f}, std={q_f.std():.4f}, norm={q_f.norm():.2f}")
        
        # Prototypes
        p_s_fg = masked_pooling(f_s, fore_mask_flat)
        p_s_bg = masked_pooling(f_s, back_mask_flat)
        p_f_fg = masked_pooling(f_f, fore_mask_flat)
        p_f_bg = masked_pooling(f_f, back_mask_flat)
        
        # Aggregate K shots
        p_s_fg = p_s_fg.view(B, K, -1, 1, 1).mean(dim=1)
        p_s_bg = p_s_bg.view(B, K, -1, 1, 1).mean(dim=1)
        p_f_fg = p_f_fg.view(B, K, -1, 1, 1).mean(dim=1)
        p_f_bg = p_f_bg.view(B, K, -1, 1, 1).mean(dim=1)
        
        print(f"\n  --- Prototype Norms ---")
        print(f"  Spatial FG prototype norm: {p_s_fg.norm():.4f}")
        print(f"  Spatial BG prototype norm: {p_s_bg.norm():.4f}")
        print(f"  Freq FG prototype norm:    {p_f_fg.norm():.4f}")
        print(f"  Freq BG prototype norm:    {p_f_bg.norm():.4f}")
        
        # Cosine similarity between FG and BG prototypes (separation quality)
        cos_sim_spatial = torch.nn.functional.cosine_similarity(p_s_fg.view(B, -1), p_s_bg.view(B, -1), dim=1)
        cos_sim_freq = torch.nn.functional.cosine_similarity(p_f_fg.view(B, -1), p_f_bg.view(B, -1), dim=1)
        print(f"\n  --- Prototype Separation (lower = better) ---")
        print(f"  Cosine(spatial_fg, spatial_bg): {cos_sim_spatial.mean().item():.4f}")
        print(f"  Cosine(freq_fg, freq_bg):       {cos_sim_freq.mean().item():.4f}")
        
        # Fused prototypes
        p_fused_fg = model.prototype_fusion(p_s_fg, p_f_fg)
        p_fused_bg = model.prototype_fusion(p_s_bg, p_f_bg)
        
        # Band weighting
        p_f_weighted_fg = model.band_weighting(p_f_fg)
        p_f_weighted_bg = model.band_weighting(p_f_bg)
        
        # Similarity maps
        spatial_sim_fg = compute_similarity_map(q_s, p_s_fg)
        spatial_sim_bg = compute_similarity_map(q_s, p_s_bg)
        fused_sim_fg = compute_similarity_map(q_s, p_fused_fg)
        fused_sim_bg = compute_similarity_map(q_s, p_fused_bg)
        freq_sim_fg = compute_similarity_map(q_f, p_f_weighted_fg)
        freq_sim_bg = compute_similarity_map(q_f, p_f_weighted_bg)
        
        print(f"\n  --- Similarity Map Statistics ---")
        print(f"  Spatial FG sim: mean={spatial_sim_fg.mean():.4f}, std={spatial_sim_fg.std():.4f}, min={spatial_sim_fg.min():.4f}, max={spatial_sim_fg.max():.4f}")
        print(f"  Spatial BG sim: mean={spatial_sim_bg.mean():.4f}, std={spatial_sim_bg.std():.4f}, min={spatial_sim_bg.min():.4f}, max={spatial_sim_bg.max():.4f}")
        print(f"  Fused FG sim:   mean={fused_sim_fg.mean():.4f}, std={fused_sim_fg.std():.4f}, min={fused_sim_fg.min():.4f}, max={fused_sim_fg.max():.4f}")
        print(f"  Fused BG sim:   mean={fused_sim_bg.mean():.4f}, std={fused_sim_bg.std():.4f}, min={fused_sim_bg.min():.4f}, max={fused_sim_bg.max():.4f}")
        print(f"  Freq FG sim:    mean={freq_sim_fg.mean():.4f}, std={freq_sim_fg.std():.4f}, min={freq_sim_fg.min():.4f}, max={freq_sim_fg.max():.4f}")
        print(f"  Freq BG sim:    mean={freq_sim_bg.mean():.4f}, std={freq_sim_bg.std():.4f}, min={freq_sim_bg.min():.4f}, max={freq_sim_bg.max():.4f}")
        
        # Contrast: FG - BG (positive = correctly discriminating)
        spatial_contrast = (spatial_sim_fg - spatial_sim_bg).mean().item()
        fused_contrast = (fused_sim_fg - fused_sim_bg).mean().item()
        freq_contrast = (freq_sim_fg - freq_sim_bg).mean().item()
        print(f"\n  --- Similarity Contrast (FG - BG, positive = discriminative) ---")
        print(f"  Spatial contrast: {spatial_contrast:.4f}")
        print(f"  Fused contrast:   {fused_contrast:.4f}")
        print(f"  Freq contrast:    {freq_contrast:.4f}")
        
        # Final fusion
        w1 = model.fusion_weight1.item()
        w2 = model.fusion_weight2.item()
        sim_fg = w2 * spatial_sim_fg + w1 * freq_sim_fg + (1 - w1 - w2) * fused_sim_fg
        sim_bg = w2 * spatial_sim_bg + w1 * freq_sim_bg + (1 - w1 - w2) * fused_sim_bg
        
        final_contrast = (sim_fg - sim_bg).mean().item()
        print(f"  Final fused contrast: {final_contrast:.4f}")


def visualize_similarity_maps(model, support_img, query_img, support_mask, target_mask, device, fold, ep_idx, save_dir="diagnostics"):
    """Visualize the individual similarity maps side by side."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    K = support_img.shape[1]
    B = support_img.shape[0]
    
    with torch.no_grad():
        s_img_flat = support_img.view(B * K, 3, support_img.shape[3], support_img.shape[4])
        fore_mask_flat = support_mask.view(B * K, 1, support_mask.shape[3], support_mask.shape[4])
        back_mask_flat = 1 - fore_mask_flat
        
        f_s = model.spatial_encoder(s_img_flat)
        f_f = model.freq_encoder(s_img_flat)
        q_s = model.spatial_encoder(query_img)
        q_f = model.freq_encoder(query_img)
        
        p_s_fg = masked_pooling(f_s, fore_mask_flat).view(B, K, -1, 1, 1).mean(dim=1)
        p_s_bg = masked_pooling(f_s, back_mask_flat).view(B, K, -1, 1, 1).mean(dim=1)
        p_f_fg = masked_pooling(f_f, fore_mask_flat).view(B, K, -1, 1, 1).mean(dim=1)
        p_f_bg = masked_pooling(f_f, back_mask_flat).view(B, K, -1, 1, 1).mean(dim=1)
        
        p_fused_fg = model.prototype_fusion(p_s_fg, p_f_fg)
        p_fused_bg = model.prototype_fusion(p_s_bg, p_f_bg)
        p_f_weighted_fg = model.band_weighting(p_f_fg)
        p_f_weighted_bg = model.band_weighting(p_f_bg)
        
        spatial_sim_fg = compute_similarity_map(q_s, p_s_fg)
        spatial_sim_bg = compute_similarity_map(q_s, p_s_bg)
        fused_sim_fg = compute_similarity_map(q_s, p_fused_fg)
        fused_sim_bg = compute_similarity_map(q_s, p_fused_bg)
        freq_sim_fg = compute_similarity_map(q_f, p_f_weighted_fg)
        freq_sim_bg = compute_similarity_map(q_f, p_f_weighted_bg)
        
        # Final prediction
        logits, _ = model(support_img, query_img, support_mask)
        preds = logits.argmax(dim=1)
        
        # Convert to numpy for plotting
        q_img_np = query_img[0].permute(1, 2, 0).cpu().numpy()
        gt_np = target_mask[0].cpu().numpy()
        pred_np = preds[0].cpu().numpy()
        
        # Upsample similarity maps to image resolution for visualization
        H, W = query_img.shape[-2:]
        
        def up(t):
            return torch.nn.functional.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        
        spatial_diff = up(spatial_sim_fg - spatial_sim_bg)
        fused_diff = up(fused_sim_fg - fused_sim_bg)
        freq_diff = up(freq_sim_fg - freq_sim_bg)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Top row: images and masks
        axes[0, 0].imshow(q_img_np)
        axes[0, 0].set_title("Query Image")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(gt_np, cmap='gray')
        axes[0, 1].set_title("Ground Truth")
        axes[0, 1].axis("off")
        
        axes[0, 2].imshow(pred_np, cmap='gray')
        axes[0, 2].set_title("Prediction")
        axes[0, 2].axis("off")
        
        axes[0, 3].axis("off")
        w1 = model.fusion_weight1.item()
        w2 = model.fusion_weight2.item()
        info_text = f"fusion_w1 (freq): {w1:.4f}\nfusion_w2 (spatial): {w2:.4f}\nfused: {1-w1-w2:.4f}\nscaler: {model.scaler.item():.2f}"
        axes[0, 3].text(0.1, 0.5, info_text, fontsize=14, family='monospace', verticalalignment='center')
        axes[0, 3].set_title("Learned Params")
        
        # Bottom row: similarity contrast maps (FG - BG)
        vmax = max(abs(spatial_diff).max(), abs(fused_diff).max(), abs(freq_diff).max())
        
        im1 = axes[1, 0].imshow(spatial_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, 0].set_title("Spatial (FG-BG)")
        axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        im2 = axes[1, 1].imshow(fused_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, 1].set_title("Fused MHA (FG-BG)")
        axes[1, 1].axis("off")
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        im3 = axes[1, 2].imshow(freq_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, 2].set_title("Freq DWT (FG-BG)")
        axes[1, 2].axis("off")
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
        
        # Combined weighted
        combined = w2 * spatial_diff + w1 * freq_diff + (1 - w1 - w2) * fused_diff
        im4 = axes[1, 3].imshow(combined, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, 3].set_title("Weighted Combined")
        axes[1, 3].axis("off")
        plt.colorbar(im4, ax=axes[1, 3], fraction=0.046)
        
        plt.suptitle(f"Fold {fold} - Episode {ep_idx+1} | Red = FG, Blue = BG", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/diag_fold_{fold}_ep_{ep_idx+1}.png", dpi=150)
        plt.show()
        plt.close()


def main():
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading dataset for diagnostics...")
    dataset = TumorDataset()
    available_classes = list(dataset.label_to_indices.keys())
    
    K = 10
    Q = 1
    diag_episodes = 3  # Run diagnostics on 3 episodes per fold
    
    for i, test_class in enumerate(available_classes):
        fold = i + 1
        unseen = [test_class]
        
        model_path = f"models/panet_fold_{fold}.pth"
        if not os.path.exists(model_path):
            print(f"Skipping fold {fold}: Model weights not found at {model_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold} | Evaluating on unseen class: {unseen}")
        print(f"{'='*60}")
        
        model = DualBranchPANet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Print model-level diagnostics
        diagnose_model(model, device, tag=f"(Fold {fold})")
        
        # Run episode-level diagnostics
        eval_sampler = EpisodicBatchSampler(dataset.label_to_indices, unseen, num_episodes=diag_episodes, k_shots=K, q_queries=Q)
        eval_loader = DataLoader(dataset, batch_sampler=eval_sampler)
        
        for ep_idx, (imgs, masks, labels) in enumerate(eval_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            support_img = imgs[:K].unsqueeze(0)
            support_mask = masks[:K].unsqueeze(0)
            query_img = imgs[K:].unsqueeze(0).squeeze(1)
            target_mask = masks[K:].squeeze(1)
            
            diagnose_episode(model, support_img, query_img, support_mask, target_mask, device, fold, ep_idx)
            visualize_similarity_maps(model, support_img, query_img, support_mask, target_mask, device, fold, ep_idx)


if __name__ == "__main__":
    main()
