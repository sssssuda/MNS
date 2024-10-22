import torch
from torch.utils.data import Dataset
from mns_model.data_process import get_LogFillterBank, load_data_frompkl, enframe, load_data_fromdat, get_segments
import numpy as np

from mns_model.utils import load_masc_data
import soundfile as sf


class SpeechMascDataset(Dataset):
    def __init__(self, path=None, split="train", nfilt=26):
        data, label = load_data_frompkl(path, split=split)

        lfb = get_LogFillterBank(data, sample_rate=8000, nfilt=nfilt)
        label = np.array(label)

        data, labels = enframe(lfb, label)
        self.data = torch.from_numpy(data).to(torch.float32).unsqueeze(2)
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class SpeechMascListDataset(Dataset):
    def __init__(self, path=None, split="train", nfilt=26, lstm=False, cut=False, n_fold=0):
        datas, labels = load_masc_data(split=split, n_fold=n_fold)
        data, label = [], []
        for wav_path, idx_label in zip(datas, labels):
            wav, sr = sf.read(wav_path)
            data.append(wav)
            oh_label = np.zeros(18)
            oh_label[idx_label] = 1
            label.append(oh_label)

        lfb = get_LogFillterBank(data, sample_rate=8000, nfilt=nfilt)
        label = np.array(label)

        if lstm:
            data, labels = get_segments(lfb, label)
        else:
            data, labels = enframe(lfb, label)

        # .unsqueeze(2) mns/cnn-frame用    # .unsqueeze(3)  xvector用    # lstm 不需要unsqueeze
        self.data = torch.from_numpy(data).to(torch.float32).unsqueeze(3)
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class SpeechEmoDataset(Dataset):
    def __init__(self, path=None, split="train", nfilt=26, lstm=False, cut=False):
        data, label = load_data_fromdat(path, split=split)

        lfb = get_LogFillterBank(data, sample_rate=16000, nfilt=nfilt)
        label = np.array(label)

        if lstm:
            data, labels = get_segments(lfb, label)
        else:
            data, labels = enframe(lfb, label)

        # .unsqueeze(2) mns用    # .unsqueeze(3)  xvector用    # lstm 不需要unsqueeze
        self.data = torch.from_numpy(data).to(torch.float32).unsqueeze(3)
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class SpeechUtteranceDataset(Dataset):
    def __init__(self, path=None, split="train", nfilt=26):
        data, label = load_data_frompkl(path, split=split)

        lfb = get_LogFillterBank(data, sample_rate=8000, nfilt=nfilt)
        label = np.array(label)

        self.data = lfb   #  torch.from_numpy(lfb).to(torch.float32)    # (utterance, frames,dim)
        self.labels = torch.from_numpy(label)    # (utterance, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
