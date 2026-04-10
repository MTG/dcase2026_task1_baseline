import random
import numpy as np
import torch
from torch.utils.data import Dataset


class HATRDataset(Dataset):
    """
    Dataset for precomputed multimodal (audio + text) embeddings.
    Hierarchical labels (2-level, use only parent and leafs if more).
    Augmentation (optional): Gaussian noise + random zeroing on embeddings.
    CSV columns are hardcoded for BST datasets, change if necessary.
    """

    def __init__(self, dataframe, aug=True, mask_pct=0.2):
        self.dataframe = dataframe
        self.aug = aug
        self.mask_pct = mask_pct

    def _rand_mask(self, emb):
        max_mask = int(emb.shape[0] * self.mask_pct)
        num_to_mask = random.randint(1, max_mask)
        mask_indices = torch.randperm(emb.shape[0])[:num_to_mask]
        mask = torch.ones_like(emb)
        mask[mask_indices] = 0.0
        return emb * mask

    def get_classes(self):
        return self.dataframe['class'].unique()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        sound_id = sample['index']
        class_name = sample['class']
        top_class_name = sample['top_class']
        class_idx = sample['class_idx']
        top_class_idx = sample['top_class_idx']
        
        emb_path = sample['audio_emb_filepath']
        emb = torch.tensor(np.load(emb_path), dtype=torch.float32)

        text_path = sample['text_emb_filepath']
        text_emb = torch.tensor(np.load(text_path), dtype=torch.float32)

        if self.aug:
            emb = emb + torch.randn_like(emb) * 0.0001
            emb = self._rand_mask(emb)
            text_emb = text_emb + torch.randn_like(text_emb) * 0.0001
            text_emb = self._rand_mask(text_emb)

        sample_data = {
            'sound_id': sound_id,
            'audio_embedding': emb,
            'text_embedding': text_emb,
            'class': class_name,
            'class_idx': class_idx,
            'top_class': top_class_name,
            'top_class_idx': top_class_idx,
        } 
        
        return sample_data
