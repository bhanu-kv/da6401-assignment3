import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import wandb

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, 
                    names=['latin', 'devanagari', 'count'])
    
    # Convert all data to strings and handle missing values
    df['latin'] = df['latin'].astype(str).str.strip()
    df['devanagari'] = df['devanagari'].astype(str).str.strip()
    
    # Remove rows with empty strings or invalid data
    df = df[(df['latin'] != 'nan') & (df['devanagari'] != 'nan')]
    
    return list(zip(df['latin'], df['devanagari']))

# Vocabulary
class Vocabulary:
    def __init__(self):
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.max_length = 0

    def build_vocab(self, data):
        chars = set()
        for src, tgt in data:
            chars.update(src)
            chars.update(tgt)
            self.max_length = max(self.max_length, len(src), len(tgt))
        for char in sorted(chars):
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

# Dataset and Loader
class TransliterationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_indices = [self.src_vocab.char2idx.get(c, 3) for c in src]
        tgt_indices = [self.tgt_vocab.char2idx.get(c, 3) for c in tgt]
        src_tensor = torch.LongTensor([1] + src_indices + [2])
        tgt_tensor = torch.LongTensor([1] + tgt_indices + [2])
        return src_tensor, tgt_tensor

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, padding_value=0)
    return src_padded, tgt_padded
