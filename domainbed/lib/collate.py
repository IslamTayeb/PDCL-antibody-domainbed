import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable

class Alphabet:
    """ESM-style alphabet for tokenizing protein sequences"""
    def __init__(self):
        self.padding_idx = 0  # Default padding index
        self.cls_idx = 1      # Start of sequence token
        self.eos_idx = 2      # End of sequence token
        self.mask_idx = 3     # Mask token
        
        # Standard amino acid vocabulary
        self.tokens = ['<pad>', '<cls>', '<eos>', '<mask>', 
                      'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
        
        # Create a mapping from tokens to indices
        self.token_to_idx = {tok: i for i, tok in enumerate(self.tokens)}
        
    def get_batch_converter(self):
        """Returns a callable batch converter function"""
        return lambda batch: self.batch_converter(batch)
        
    def batch_converter(self, batch):
        """
        Convert a batch of sequences to token indices
        
        Args:
            batch: List of tuples (name, sequence)
            
        Returns:
            batched_ids: List of token indices
        """
        batch_size = len(batch)
        names = [b[0] for b in batch]
        sequences = [b[1] for b in batch]
        
        # Map each amino acid to its index, adding cls/eos tokens
        batched_ids = []
        for seq in sequences:
            ids = [self.cls_idx]  # Start with <cls> token
            for aa in seq:
                if aa in self.token_to_idx:
                    ids.append(self.token_to_idx[aa])
                else:
                    # Use mask token for unknown amino acids
                    ids.append(self.mask_idx)
            ids.append(self.eos_idx)  # End with <eos> token
            batched_ids.append(torch.tensor(ids, dtype=torch.long))
        
        # Pad sequences to the same length
        max_len = max(len(ids) for ids in batched_ids)
        batched_ids = [F.pad(ids, (0, max_len - len(ids)), 
                          value=self.padding_idx) for ids in batched_ids]
        
        return names, sequences, torch.stack(batched_ids)


class PaddCollator:
    """Collate and pad sequences of varying lengths"""
    def __init__(self, pad_value=0):
        self.pad_value = pad_value
        
    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (x, y)
            
        Returns:
            Padded x, stacked y
        """
        xs = [item[0] for item in batch]
        ys = [item[1] for item in batch]
        
        # Get max sequence length
        max_len = max(x.shape[0] for x in xs)
        
        # Pad sequences
        padded_xs = []
        for x in xs:
            padding = [(0, 0) for _ in range(x.ndim)]
            padding[0] = (0, max_len - x.shape[0])
            padded_x = F.pad(x, [p for pad in reversed(padding) for p in pad], 
                           value=self.pad_value)
            padded_xs.append(padded_x)
            
        return torch.stack(padded_xs), torch.stack(ys)


def pad_x(batch, padding_idx=0):
    """
    Pad a batch of sequences to the same length
    
    Args:
        batch: List of tuples (x, y)
        padding_idx: Value to use for padding
        
    Returns:
        List of tuples with padded x
    """
    max_len = max(x.shape[1] for x, _ in batch)
    padded_batch = []
    
    for x, y in batch:
        if x.shape[1] < max_len:
            padding = (0, 0, 0, max_len - x.shape[1])
            padded_x = F.pad(x, padding, value=padding_idx)
            padded_batch.append((padded_x, y))
        else:
            padded_batch.append((x, y))
            
    return padded_batch


def rn_collate(batch):
    """
    Collate function for regular networks
    
    Args:
        batch: List of tuples (x, y)
        
    Returns:
        Collated batch with stacked x and y
    """
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    
    # Check if all xs have the same shape
    if all(x.shape == xs[0].shape for x in xs):
        return torch.stack(xs), torch.stack(ys)
    else:
        # If shapes differ, need to pad
        return PaddCollator()(batch)


def esm_collate(batch, x_collate_fn=None):
    """
    Collate function for ESM models
    
    Args:
        batch: List of tuples (x, y)
        x_collate_fn: Optional function to convert x
        
    Returns:
        Collated batch with x processed for ESM
    """
    if x_collate_fn is not None:
        # Convert protein sequences using the provided function
        x = [item[0] for item in batch]
        y = torch.stack([item[1] for item in batch])
        
        # Process sequences with x_collate_fn
        processed_x = x_collate_fn(x)
        return processed_x, y
    else:
        # Fall back to regular collate
        return rn_collate(batch)