import json
import argparse
import os

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def clean_local_imports(code_str):
    """Strips local module imports and specific entry point markers when concatenating into the notebook cell."""
    lines = code_str.split('\n')
    cleaned = []
    skip_next = False
    for line in lines:
        if skip_next:
            if line.strip() == '' or line.startswith('    ') or line.startswith('train_freqfss(') or line.startswith('config=Config()') or line.startswith('dataset_name=') or line.startswith('make_episode_loaders_fn='):
                continue
            skip_next = False
            
        stripped = line.strip()
        
        if stripped.startswith('from imports import'):
            continue
        if stripped.startswith('from model import'):
            continue
        if stripped.startswith('from train_eval import'):
            continue
            
        if stripped == "if __name__ == '__main__':":
            skip_next = True
            continue
            
        cleaned.append(line)
        
    return '\n'.join(cleaned)

def create_notebook(dataset_name):
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
        
    dataset_file = f"pipeline/dataset_{dataset_name}.py"
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} not found. Available datasets: isic2016, busi, kvasir, brisc2025")
        return

    dataset_display_name = {
        "isic2016": "ISIC 2016",
        "busi": "BUSI",
        "kvasir": "Kvasir-SEG",
        "brisc2025": "BRISC 2025"
    }.get(dataset_name, dataset_name.upper())

    # Cell 1: Description
    add_markdown(f"# FreqFSS Few-Shot Segmentation\n**Dataset**: {dataset_display_name}\n\nThis notebook is automatically generated from modular components.")

    # Cell 2: All-in-one Code Cell
    imports_code = read_file("pipeline/imports.py")
    dataset_code = clean_local_imports(read_file(dataset_file))
    model_code = clean_local_imports(read_file("pipeline/model.py"))
    train_eval_code = clean_local_imports(read_file("pipeline/train_eval.py"))
    
    combined_code = "\n\n".join([imports_code, dataset_code, model_code, train_eval_code])
    
    # Append the execution command to the notebook cell so it runs out-of-the-box in Kaggle
    execution_code = f"""

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    train_freqfss(
        config=Config(),
        dataset_name="{dataset_display_name}",
        make_episode_loaders_fn=make_episode_loaders
    )
"""
    
    combined_code += execution_code
    
    add_code(combined_code)

    out_file = f'freqfss_{dataset_name}.ipynb'
    with open(out_file, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"Notebook {out_file} generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kaggle Notebook for a dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (isic2016, busi, kvasir, brisc2025)")
    args = parser.parse_args()
    create_notebook(args.dataset)
