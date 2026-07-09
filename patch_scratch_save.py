import glob

patch_old = """            # Save best checkpoint
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
                print(f"  ✓ Best model saved (Val Dice {best_val_dice:.4f})")"""

patch_new = """            # Save best checkpoint
            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                if getattr(config, 'SAVE_MODEL', True):
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
                    print(f"  ✓ Best model identified (Val Dice {best_val_dice:.4f}) [Skipping save for local run]")"""

for f in glob.glob("scratch_extracted/freqfss_*.py"):
    with open(f, "r") as file:
        content = file.read()
    
    if "getattr(config, 'SAVE_MODEL', True)" not in content:
        content = content.replace(patch_old, patch_new)
        with open(f, "w") as file:
            file.write(content)
        print(f"Updated {f}")
