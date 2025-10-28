from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from glob import glob
from typing import Literal, List
from pedalboard.io import ReadableAudioFile
from torch.utils.data import Dataset
from tfcrnn.utils import mkpath
from tfcrnn.config import Config




class FMADataset(Dataset):

  def __init__(
    self,
    split: Literal['training', 'validation', 'test'],
    paths: list[str],
    labels: list[str],
    config: Config,
  ):
        self.split = split
        self.config = config
        self.paths: list[str] = paths
        self.labels: list[str] = labels
        self.sample_rate = getattr(config, 'sample_rate', 16000)
        self.input_size = getattr(config, 'input_size', 1323000)

  # def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
  #     path = self.paths[i]
  #     label = self.labels[i]

  #     with ReadableAudioFile(path).resampled_to(self.config.sample_rate) as af:
  #       assert af.samplerate == self.config.sample_rate, (
  #         f'The configured sampling rate is {self.config.sample_rate}, '
  #         f'but got {af.samplerate} from: {path}'
  #       )
  #       x = af.read(self.config.input_size)
  #       if x.size == 0:
  #           print(f"Warning: No audio data read from: {path}. Skipping item {i}.")
  #           x = np.zeros((1, self.config.input_size), dtype=np.float32)

  #       if x.ndim < 2 or x.shape[0] == 0:
  #           print(f"Warning: Unexpected shape {x.shape} for audio data from: {path}. Expected at least 2 dimensions.")
  #           x = np.zeros((1, self.config.input_size), dtype=np.float32)

  #       # select one channel only
  #       x = x[0, :]
  #       x = x.squeeze()
  #       x = torch.from_numpy(x)
  #       #x = x / 0.0860

  #     if len(x) < self.config.input_size:
  #       # Pad with zeros if audio is shorter than 16000 (1 sec).
  #       pad_size = self.config.input_size - len(x)
  #       x = F.pad(x.float(), (pad_size // 2, pad_size // 2 + pad_size % 2), mode='constant', value=0)
      
  #     x = x.unsqueeze(0).float()
  #     assert x.dtype == torch.float32, f"x is {x.dtype} (expected float32)"
      
  #     assert x.shape == (1, self.config.input_size), (
  #       f'The processed waveform should have a shape of {(1, self.config.input_size)}, '
  #       f'but got {x.shape}'
  #     )

  #     return x, label
  def __getitem__(self, i: int):
    path = self.paths[i]
    label = self.labels[i]

    try:
        with ReadableAudioFile(path).resampled_to(self.config.sample_rate) as af:
            x = af.read(self.config.input_size)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return self.__getitem__((i + 1) % len(self.paths))  # skip to next sample

    # If empty or invalid
    if x.size == 0 or x.ndim < 2:
        print(f"Skipping empty audio {path}")
        return self.__getitem__((i + 1) % len(self.paths))

    x = x[0, :self.config.input_size].astype(np.float32, copy=False)
    x = torch.from_numpy(x).float()

    if len(x) < self.config.input_size:
        pad_size = self.config.input_size - len(x)
        x = F.pad(x, (pad_size // 2, pad_size // 2 + pad_size % 2), value=0.0)

    x = x.unsqueeze(0).contiguous().float()

    assert x.dtype == torch.float32, f"dtype mismatch at {path}: {x.dtype}"
    return x, label


  def __len__(self):
      return len(self.paths)


def load_audio_paths(
  dataset_dir: str,
  split: Literal['train', 'valid', 'test'],
) -> List[str]:
  if split != 'test':
    with open(mkpath(dataset_dir, 'validation_list.txt')) as f:
      valid_paths = f.read().splitlines()
      valid_paths = [mkpath(dataset_dir, path) for path in valid_paths]
      valid_paths.sort()
  
  if split == 'valid':
    return valid_paths
  
  with open(mkpath(dataset_dir, 'testing_list.txt')) as f:
    test_paths = f.read().splitlines()
    test_paths = [mkpath(dataset_dir, path) for path in test_paths]
    test_paths.sort()
  
  if split == 'test':
    return test_paths
  
  audio_paths = glob(mkpath(dataset_dir, '*/*.wav'))
  noise_paths = glob(mkpath(dataset_dir, '_background_noise_/*.wav'))
  
  # Remove validation, test set, and noises from the training set.
  train_paths = list(set(audio_paths) - set(valid_paths) - set(test_paths) - set(noise_paths))
  train_paths.sort()
  
  return train_paths



def load_audio_paths_fma(
    tracks_df: pd.DataFrame,
    audio_root_dir: Path,
    name_to_index_map: dict,
    split: Literal['training', 'validation', 'test'],
) -> Tuple[List[str], List[int]]:

    split_tracks = tracks_df.loc[(tracks_df['set', 'split'] == split)]
    genre_names = split_tracks['track', 'genre_top'].fillna('Unknown')
    labels = genre_names.apply(lambda name: name_to_index_map[name]).tolist()

    paths = []

    for track_id in split_tracks.index:
        # FMA naming convention:
        # The file is named: [Track ID].mp3 (e.g., 000002.mp3)
        # The parent folder is named: [First 3 digits of Track ID] (e.g., 000)

        track_id_str = str(track_id).zfill(6) # e.g., 2 -> "000002"
        folder_id_str = track_id_str[:3]      # e.g., "000002" -> "000"

        # Construct the full path: fma_small/000/000002.mp3
        file_path = audio_root_dir + folder_id_str + f'/{track_id_str}.mp3'

        paths.append(str(file_path))

    return paths, labels

if __name__ == '__main__':
  from tqdm import tqdm
  from torch.utils.data import DataLoader
  
  print('=> Start sanity check for the dataset')
  config = Config()
  config.parse_cli()
  config.print()
  
  splits = ['train', 'valid', 'test']
  datasets = [SpeechCommandsDataset(split, config) for split in splits]
  for dataset in datasets:
    loader = DataLoader(dataset, config.batch_size, shuffle=False, drop_last=False, num_workers=0)
    for x, y in tqdm(loader, desc=dataset.split):
      pass
