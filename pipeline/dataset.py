import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T

class TumorDataset(Dataset):
    def __init__(self, manifest_path="../Data/brisc2025/manifest.csv", data_root="../Data/brisc2025/", image_size=(256, 256)):
        self.data_root = data_root
        self.image_size = image_size
        
        # We will parse the manifest.csv and store valid samples
        self.samples = []
        self.label_to_indices = {}
        
        df = pd.read_csv(manifest_path)
        
        # Filter for segmentation task and mask entries
        df_seg = df[(df['task'] == 'segmentation') & (df['is_mask'] == True)]
        
        # Exclude 'no_tumor' as requested
        df_seg = df_seg[df_seg['tumor_label'] != 'no_tumor']
        
        for idx, row in df_seg.iterrows():
            mask_path = row['relative_path'].replace('\\', '/')
            img_path = row['linked_image'].replace('\\', '/')
            label = row['tumor_label']
            
            self.samples.append({
                'image': os.path.join(self.data_root, img_path),
                'mask': os.path.join(self.data_root, mask_path),
                'label': label
            })
            
            # Group by label
            curr_idx = len(self.samples) - 1
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(curr_idx)
            
        self.img_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
        ])
        
        self.mask_transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image']).convert("RGB")
        mask = Image.open(sample['mask']).convert("L")  # Greyscale for binary mask
        
        img_t = self.img_transform(img)
        mask_t = self.mask_transform(mask)
        
        # Binarize mask just in case
        mask_t = (mask_t > 0).float()
        
        return img_t, mask_t, sample['label']


class EpisodicBatchSampler(Sampler):
    def __init__(self, label_to_indices, seen_classes, num_episodes, k_shots, q_queries):
        self.label_to_indices = label_to_indices
        self.seen_classes = seen_classes
        self.num_episodes = num_episodes
        self.k_shots = k_shots
        self.q_queries = q_queries

    def __iter__(self):
        for _ in range(self.num_episodes):
            # Sample a class
            sampled_class = random.choice(self.seen_classes)
            
            # Get indices for this class
            class_indices = self.label_to_indices[sampled_class]
            
            # Sample K+Q items without replacement
            sampled_indices = random.sample(class_indices, self.k_shots + self.q_queries)
            
            yield sampled_indices

    def __len__(self):
        return self.num_episodes
