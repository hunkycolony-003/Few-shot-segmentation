import glob
import numpy as np

patch = """
def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    \"\"\"Compute 95th percentile Hausdorff Distance for a single 2D image.\"\"\"
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

metrics_body_old = """    return dict(dice=float(dice), iou=float(iou),
                sensitivity=float(sens), specificity=float(spec))"""
metrics_body_new = """    hd95_list = [compute_hd95(preds[i], tgts[i]) for i in range(preds.shape[0])]
    hd95 = float(np.mean(hd95_list)) if hd95_list else 0.0

    return dict(dice=float(dice), iou=float(iou),
                sensitivity=float(sens), specificity=float(spec), hd95=float(hd95))"""

imports_old = """from torch.utils.data import Dataset, DataLoader

import albumentations as A"""

imports_new = """from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

import albumentations as A"""

print_val_old = """                f"Spec {val_metrics['specificity']:.4f}"
            )"""
print_val_new = """                f"Spec {val_metrics['specificity']:.4f} "
                f"HD95 {val_metrics['hd95']:.4f}"
            )"""

print_test_old = """    print(f"  Specificity : {test_metrics['specificity']:.4f}")
    print(f"{'═'*65}\\n")"""
print_test_new = """    print(f"  Specificity : {test_metrics['specificity']:.4f}")
    print(f"  HD95        : {test_metrics['hd95']:.4f}")
    print(f"{'═'*65}\\n")"""

for f in glob.glob("scratch_extracted/freqfss_*.py"):
    with open(f, "r") as file:
        content = file.read()
    
    if "from scipy.ndimage import binary_erosion" not in content:
        content = content.replace("def compute_metrics(pred_probs: np.ndarray, targets: np.ndarray) -> dict:", patch)
        content = content.replace(metrics_body_old, metrics_body_new)
        content = content.replace(imports_old, imports_new)
        content = content.replace(print_val_old, print_val_new)
        content = content.replace(print_test_old, print_test_new)
        
        with open(f, "w") as file:
            file.write(content)
        print(f"Updated {f}")

