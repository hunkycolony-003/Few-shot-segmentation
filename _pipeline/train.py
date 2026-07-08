import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import TumorDataset, EpisodicBatchSampler
from dual_branch_panet import DualBranchPANet

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup dataset
    print("Loading dataset...")
    dataset = TumorDataset()
    
    # Ensure there are enough samples per class
    available_classes = list(dataset.label_to_indices.keys())
    print(f"Available classes: {available_classes}")
    
    # 3-Fold Cross-Validation Setup
    K = 10  # 10-shot
    Q = 1   # 1 query
    
    epochs = 1  # Kept low for local testing
    train_episodes = 50
    eval_episodes = 50
    
    folds = []
    for i, test_class in enumerate(available_classes):
        seen_classes = [c for c in available_classes if c != test_class]
        folds.append({'fold': i+1, 'seen': seen_classes, 'unseen': [test_class]})
        
    print(f"Starting 3-Fold Cross Validation...")
    
    for fold_info in folds:
        fold = fold_info['fold']
        seen = fold_info['seen']
        unseen = fold_info['unseen']
        
        print(f"\n--- Fold {fold} ---")
        print(f"Training on: {seen}")
        print(f"Evaluating on: {unseen}")
        
        model = DualBranchPANet().to(device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        
        train_sampler = EpisodicBatchSampler(dataset.label_to_indices, seen, num_episodes=train_episodes, k_shots=K, q_queries=Q)
        train_loader = DataLoader(dataset, batch_sampler=train_sampler)
        
        # Training
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for ep_idx, (imgs, masks, labels) in enumerate(train_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                
                # Split K support and Q query
                support_img = imgs[:K].unsqueeze(0)   # (1, K, 3, H, W)
                support_mask = masks[:K].unsqueeze(0) # (1, K, 1, H, W)
                query_img = imgs[K:].unsqueeze(0)     # (1, Q, 3, H, W) -> wait, Q=1 so (1, 3, H, W)
                query_img = query_img.squeeze(1)      # (1, 3, H, W)
                query_mask = masks[K:]                # (Q, 1, H, W)
                target_mask = query_mask.squeeze(1)   # (1, H, W)
                
                optimizer.zero_grad()
                logits, loss = model(support_img, query_img, support_mask, target=target_mask)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                if (ep_idx + 1) % 5 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}] Episode [{ep_idx+1}/{train_episodes}] Loss: {loss.item():.4f}")
                    
        # Save the model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/panet_fold_{fold}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
