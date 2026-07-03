import json

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_markdown(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + '\n' for line in source.split('\n')]
        })

    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in source.split('\n')]
        })

    # Cell 1: Configuration
    add_markdown("# 1. Configuration & Imports\nSetup Kaggle dataset path and hyperparameters.")
    config_code = """import os
import random
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
import torchvision.models as models

# --- CONFIGURATION ---
DATA_ROOT = "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025"  # Update if Kaggle mounts it differently
MANIFEST_PATH = "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/manifest.csv"

# Episodic Settings
K_SHOTS = 10
Q_QUERIES = 1
TRAIN_EPISODES_PHASE_1 = 70   # Episodes to run with frozen backbone
TRAIN_EPISODES_PHASE_2 = 300  # Episodes to run with unfrozen backbone
EVAL_EPISODES = 10            # Episodes to run during evaluation

# Learning Rates
LR_FROZEN = 5e-5      # Higher LR when backbone is frozen
LR_UNFROZEN = 1e-6    # 10x lower LR for fine-tuning the un-frozen backbone
gamma = 0.8

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")"""
    add_code(config_code)

    # Cell 2: Dataset
    add_markdown("# 2. Dataset & Sampler")
    dataset_code = read_file("pipeline/dataset.py").replace('import os', '').replace('import random', '').replace('import pandas as pd', '').replace('from PIL import Image', '').replace('from torch.utils.data import Dataset, Sampler', '').replace('import torchvision.transforms as T', '')
    # Replace default paths
    dataset_code = dataset_code.replace('manifest_path="../Data/brisc2025/manifest.csv"', 'manifest_path=MANIFEST_PATH')
    dataset_code = dataset_code.replace('data_root="../Data/brisc2025/"', 'data_root=DATA_ROOT')
    add_code(dataset_code.strip())

    # Cell 3: Model
    add_markdown("# 3. Dual-Branch PANet Architecture")
    model_code = read_file("pipeline/dual_branch_panet.py")
    # Clean up imports
    model_code = model_code.replace('import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torchvision.models as models\n', '')
    # Remove the `if __name__ == "__main__":` usage block at the bottom
    model_code = model_code.split('if __name__ == "__main__":')[0].strip()
    
    # Add unfreeze_backbone method to DualBranchPANet
    model_code = model_code.replace(
        '        return logits, loss',
        '        return logits, loss\n\n    def unfreeze_backbone(self):\n        """Unfreeze the ResNet50 backbone for Phase 2 fine-tuning."""\n        for param in self.spatial_encoder.backbone.parameters():\n            param.requires_grad = True\n        print("ResNet50 backbone unfrozen.")'
    )
    add_code(model_code)

    # Cell 4: Training & Evaluation Functions
    add_markdown("# 4. Phased Training Loop & Evaluation Utilities")
    train_eval_code = """
def save_visualization(support_img, query_img, gt_mask, pred_mask, fold, episode, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)
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
    plt.show()  # Show inline in Jupyter
    plt.close()

def run_training_phase(model, train_loader, optimizer, phase_episodes, phase_name):
    print(f"--- Starting {phase_name} ({phase_episodes} episodes) ---")
    model.train()
    total_loss = 0.0
    for ep_idx, (imgs, masks, labels) in enumerate(train_loader):
        if ep_idx >= phase_episodes:
            break
            
        imgs, masks = imgs.to(device), masks.to(device)
        support_img = imgs[:K_SHOTS].unsqueeze(0)
        support_mask = masks[:K_SHOTS].unsqueeze(0)
        query_img = imgs[K_SHOTS:].unsqueeze(0).squeeze(1)
        target_mask = masks[K_SHOTS:].squeeze(1)
        
        optimizer.zero_grad()
        logits, loss = model(support_img, query_img, support_mask, target=target_mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (ep_idx + 1) % 10 == 0:
            print(f"  {phase_name} - Episode [{ep_idx+1}/{phase_episodes}] Loss: {loss.item():.4f}")
"""
    add_code(train_eval_code.strip())

    # Cell 5: Cross Validation Execution
    add_markdown("# 5. Execute 3-Fold Cross-Validation")
    exec_code = """
dataset = TumorDataset(manifest_path=MANIFEST_PATH, data_root=DATA_ROOT)
available_classes = list(dataset.label_to_indices.keys())
print(f"Available classes: {available_classes}")

folds = []
for i, test_class in enumerate(available_classes):
    seen_classes = [c for c in available_classes if c != test_class]
    folds.append({'fold': i+1, 'seen': seen_classes, 'unseen': [test_class]})

for fold_info in folds:
    fold = fold_info['fold']
    seen = fold_info['seen']
    unseen = fold_info['unseen']
    
    print(f"\\n=========================================")
    print(f"FOLD {fold} | Training on: {seen} | Evaluating on: {unseen}")
    print(f"=========================================")
    
    # Initialize Model (Backbone is frozen by default)
    model = DualBranchPANet().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Setup Phase 1 Sampler (50 episodes)
    train_sampler_p1 = EpisodicBatchSampler(dataset.label_to_indices, seen, num_episodes=TRAIN_EPISODES_PHASE_1, k_shots=K_SHOTS, q_queries=Q_QUERIES)
    train_loader_p1 = DataLoader(dataset, batch_sampler=train_sampler_p1)
    
    # Phase 1 Optimizer (LR = 1e-4)
    optimizer_p1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FROZEN)
    
    # Run Phase 1
    run_training_phase(model, train_loader_p1, optimizer_p1, TRAIN_EPISODES_PHASE_1, "Phase 1 (Frozen Backbone)")
    
    # Setup Phase 2
    if isinstance(model, nn.DataParallel):
        model.module.unfreeze_backbone()
    else:
        model.unfreeze_backbone()
    train_sampler_p2 = EpisodicBatchSampler(dataset.label_to_indices, seen, num_episodes=TRAIN_EPISODES_PHASE_2, k_shots=K_SHOTS, q_queries=Q_QUERIES)
    train_loader_p2 = DataLoader(dataset, batch_sampler=train_sampler_p2)
    
    # Phase 2 Optimizer (LR = 1e-5, updating all parameters now)
    optimizer_p2 = optim.AdamW(model.parameters(), lr=LR_UNFROZEN)
    
    # Run Phase 2
    run_training_phase(model, train_loader_p2, optimizer_p2, TRAIN_EPISODES_PHASE_2, "Phase 2 (Unfrozen Backbone)")
    
    # Save Fold Model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/panet_fold_{fold}.pth"
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}\\n")
    
    # Evaluation
    print(f"--- Evaluating Fold {fold} ---")
    model.eval()
    eval_sampler = EpisodicBatchSampler(dataset.label_to_indices, unseen, num_episodes=EVAL_EPISODES, k_shots=K_SHOTS, q_queries=Q_QUERIES)
    eval_loader = DataLoader(dataset, batch_sampler=eval_sampler)
    
    iou_list, dice_list = [], []
    
    with torch.no_grad():
        for ep_idx, (imgs, masks, labels) in enumerate(eval_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            support_img = imgs[:K_SHOTS].unsqueeze(0)
            support_mask = masks[:K_SHOTS].unsqueeze(0)
            query_img = imgs[K_SHOTS:].unsqueeze(0).squeeze(1)
            target_mask = masks[K_SHOTS:].squeeze(1)
            
            logits, _ = model(support_img, query_img, support_mask)
            preds = logits.argmax(dim=1)
            
            pred_fg = (preds == 1).float()
            target_fg = target_mask.float()
            
            intersection = (pred_fg * target_fg).sum().item()
            union = pred_fg.sum().item() + target_fg.sum().item() - intersection
            
            iou = intersection / (union + 1e-5)
            dice = (2.0 * intersection) / (pred_fg.sum().item() + target_fg.sum().item() + 1e-5)
            
            iou_list.append(iou)
            dice_list.append(dice)
            
            if ep_idx < 3:
                save_visualization(support_img.squeeze(0), query_img, target_mask, preds, fold, ep_idx+1)
                
    mean_iou = sum(iou_list) / len(iou_list)
    mean_dice = sum(dice_list) / len(dice_list)
    print(f"Fold {fold} Results -> mIoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}\\n")
"""
    add_code(exec_code.strip())

    with open('few_shot_kaggle.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

    print("Notebook few_shot_kaggle.ipynb generated successfully!")

if __name__ == "__main__":
    create_notebook()
