import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import TumorDataset, EpisodicBatchSampler
from dual_branch_panet import DualBranchPANet
from train import get_device

def save_visualization(support_img, query_img, gt_mask, pred_mask, fold, episode, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    
    # support_img is (K, 3, H, W). Let's take the first one.
    s_img = support_img[0].permute(1, 2, 0).cpu().numpy()
    q_img = query_img[0].permute(1, 2, 0).cpu().numpy()
    gt = gt_mask[0].cpu().numpy()
    pred = pred_mask[0].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(s_img)
    axes[0].set_title("Support Image (1 of K)")
    axes[0].axis("off")
    
    axes[1].imshow(q_img)
    axes[1].set_title("Query Image")
    axes[1].axis("off")
    
    axes[2].imshow(gt, cmap='gray')
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis("off")
    
    axes[3].imshow(pred, cmap='gray')
    axes[3].set_title("Predicted Mask")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fold_{fold}_ep_{episode}.png")
    plt.close()

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading dataset for evaluation...")
    dataset = TumorDataset()
    available_classes = list(dataset.label_to_indices.keys())
    
    K = 10  # 10-shot
    Q = 1   # 1 query
    eval_episodes = 5
    
    for i, test_class in enumerate(available_classes):
        fold = i + 1
        unseen = [test_class]
        
        model_path = f"models/panet_fold_{fold}.pth"
        if not os.path.exists(model_path):
            print(f"Skipping fold {fold}: Model weights not found at {model_path}")
            continue
            
        print(f"\n--- Evaluating Fold {fold} ---")
        print(f"Evaluating on: {unseen}")
        
        # Load the model
        model = DualBranchPANet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        eval_sampler = EpisodicBatchSampler(dataset.label_to_indices, unseen, num_episodes=eval_episodes, k_shots=K, q_queries=Q)
        eval_loader = DataLoader(dataset, batch_sampler=eval_sampler)
        
        iou_list, dice_list = [], []
        
        with torch.no_grad():
            for ep_idx, (imgs, masks, labels) in enumerate(eval_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                
                support_img = imgs[:K].unsqueeze(0)
                support_mask = masks[:K].unsqueeze(0)
                query_img = imgs[K:].unsqueeze(0).squeeze(1)
                target_mask = masks[K:].squeeze(1)
                
                logits, _ = model(support_img, query_img, support_mask)
                preds = logits.argmax(dim=1)  # (1, H, W)
                
                # Compute IoU and Dice for foreground (class 1)
                pred_fg = (preds == 1).float()
                target_fg = target_mask.float()
                
                intersection = (pred_fg * target_fg).sum().item()
                union = pred_fg.sum().item() + target_fg.sum().item() - intersection
                
                iou = intersection / (union + 1e-5)
                dice = (2.0 * intersection) / (pred_fg.sum().item() + target_fg.sum().item() + 1e-5)
                
                iou_list.append(iou)
                dice_list.append(dice)
                
                # Save visualization for a few episodes
                if ep_idx < 3:
                    save_visualization(support_img.squeeze(0), query_img, target_mask, preds, fold, ep_idx+1)
        
        mean_iou = sum(iou_list) / len(iou_list)
        mean_dice = sum(dice_list) / len(dice_list)
        print(f"Fold {fold} Results -> mIoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}")

if __name__ == "__main__":
    main()
