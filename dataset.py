import numpy as np
import hparams
import pathlib
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import tqdm
from pytorch_lightning import LightningDataModule

import melgan.modules

def readwavfile(path):
    data,_ = librosa.load(path,sr=hparams.sr)
    data = data.astype(np.float32)
    return data

def savewavfile(path,data):
    sf.write(path, data, hparams.sr, subtype="PCM_16")
    pass

def read_wavs(path_x, path_y):
    p_x = pathlib.Path(path_x)
    p_y = pathlib.Path(path_y)
    data_x, data_y = [], []

    for file in tqdm.tqdm([file for file in p_x.iterdir()], bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]", desc="Loading X"):
        data_x.append(readwavfile(file))
    
    for file in tqdm.tqdm([file for file in p_y.iterdir()], bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]", desc="Loading Y"):
        data_y.append(readwavfile(file))

    return data_x, data_y


class MelDataset(Dataset):
    def __init__(self, path_x, path_y):
        tmp_x, tmp_y = read_wavs(path_x, path_y)
        self.data_x, self.data_y = [], []
        self.wav2mel = melgan.modules.Wav2Mel()

        for x in tmp_x:
            tmp = self.wav2mel(x)
            if tmp.shape[1] > hparams.learn_length:
                self.data_x.append(tmp)
        for y in tmp_y:
            tmp = self.wav2mel(y)
            if tmp.shape[1] > hparams.learn_length:
                self.data_y.append(tmp)
        
    def __getitem__(self, index):
        x_index = index % len(self.data_x)
        y_index = index // len(self.data_x)
        x, y = self.data_x[x_index], self.data_y[y_index]
        return x, y

    def __len__(self):
        return len(self.data_x)*len(self.data_y)

class MelDataModule(LightningDataModule):

    def __init__(self, data_dir_x, data_dir_y, batch_size):
        super().__init__()
        self.data_dir_x = data_dir_x
        self.data_dir_y = data_dir_y
        self.batch_size = batch_size

    def setup(self, stage = None):
        self.train_data = MelDataset(self.data_dir_x, self.data_dir_y)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=32, shuffle=True)
