from __future__ import annotations

import os
import torch
import torch.nn.functional as F
import pandas as pd

from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from tfcrnn.config import Config
from tfcrnn.dataset import FMADataset, load_audio_paths_fma
from .base_runner import BaseRunner

class FMARunner(BaseRunner):
  def __init__(
    self,
    config: Config,
    checkpoint_path: str = None,
  ):
    super().__init__(config, checkpoint_path)

    tracks = pd.read_csv(config.metadata_dir+'tracks.csv', index_col=0, header=[0, 1])
    genres = pd.read_csv(config.metadata_dir+'genres.csv', index_col=0)
    small_tracks = tracks.loc[(tracks['set', 'subset'] == 'small')]
    genre_genres = small_tracks['track', 'genre_top'].fillna('Unknown')
    UNIQUE_GENRES = sorted(genre_genres.unique().tolist())
    NAME2IDX = {name: i for i, name in enumerate(UNIQUE_GENRES)}

    train_paths, train_labels = load_audio_paths_fma(small_tracks, config.dataset_dir, NAME2IDX, 'training')
    valid_paths, valid_labels = load_audio_paths_fma(small_tracks, config.dataset_dir, NAME2IDX, 'validation')
    test_paths, test_labels = load_audio_paths_fma(small_tracks, config.dataset_dir, NAME2IDX, 'test')
    
    self.dataset_train = FMADataset('training', train_paths, train_labels, config)
    self.dataset_valid = FMADataset('validation', valid_paths, valid_labels, config)
    self.dataset_test = FMADataset('test', test_paths, test_labels, config)

    num_workers = min(32, os.cpu_count())

    self.loader_train = DataLoader(
      self.dataset_train, config.batch_size,
      num_workers=num_workers, shuffle=True, drop_last=True
    )
    self.loader_valid = DataLoader(
      self.dataset_valid, config.batch_size,
      num_workers=num_workers, shuffle=False, drop_last=False
    )
    self.loader_test = DataLoader(
      self.dataset_test, config.batch_size,
      num_workers=num_workers, shuffle=False, drop_last=False
    )
  
  def train(self):
    self.model.train()
    
    sum_loss, sum_acc, n = 0., 0., 0.
    pbar = tqdm(self.loader_train)
    for x, y in pbar:
      x = x.to(self.device)
      y = y.to(self.device)
      
      if self.config.skeleton == 'cnn':
        loss, logit = self.compute_loss_cnn(x, y)
        acc = self.accuracy(logit, y)
      else:
        loss, logits = self.compute_loss_crnn(x, y)
        acc = self.accuracy(logits[-1], y)
      
      # Optimize.
      loss.backward()
      self.optimizer.step()
      self.optimizer.zero_grad()
      
      # Log metrics.
      batch_size = y.shape[0]
      sum_loss += batch_size * loss.item()
      sum_acc += batch_size * acc
      n += batch_size
      self.total_trained_samples += batch_size
      self.total_trained_steps += 1
      
      pbar.set_postfix(OrderedDict([
        (f'loss_train', f'{sum_loss / n:.4f}'),
        (f'acc_train', f'{sum_acc / n:.4f}'),
      ]))
    
    epoch_loss = sum_loss / len(self.dataset_train)
    epoch_acc = sum_acc / len(self.dataset_train)
    
    return epoch_loss, {'score': epoch_acc}
  
  def eval(self, loader):
    self.model.eval()
    
    sum_loss, sum_acc, n = 0., 0., 0.
    pbar = tqdm(loader)
    with torch.no_grad():
      for x, y in pbar:
        x = x.to(self.device)
        y = y.to(self.device)
        
        if self.config.skeleton == 'cnn':
          loss, logit = self.compute_loss_cnn(x, y)
          acc = self.accuracy(logit, y)
        else:
          logits, _ = self.model(x)
          loss = F.cross_entropy(logits[-1], y)
          acc = self.accuracy(logits[-1], y)
        
        # Log metrics.
        batch_size = y.shape[0]
        sum_loss += batch_size * loss.item()
        sum_acc += batch_size * acc
        n += batch_size
        
        pbar.set_postfix(OrderedDict([
          (f'loss_{loader.dataset.split}', f'{sum_loss / n:.4f}'),
          (f'acc_{loader.dataset.split}', f'{sum_acc / n:.4f}'),
        ]))
    
    epoch_loss = sum_loss / len(loader.dataset)
    epoch_acc = sum_acc / len(loader.dataset)
    
    return epoch_loss, {'score': epoch_acc}
