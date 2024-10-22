import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, float32, long
import soundfile as sf
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, csv_path, label_dict, audio_base_path='./data/audios'):
        # 读取所有人的语音和对应的人标签

        self.data = pd.read_csv(csv_path)
        self.label_dict = label_dict
        self.audio_base_path = audio_base_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_base_path, self.data.loc[idx, 'file'])

        arr = np.zeros(3)

        arr[self.label_dict[self.data.loc[idx, 'speaker']]] = 1
        try:
            speech_array, _ = sf.read(audio_path)
        except Exception as e:
            raise IOError(f"An error occurred when reading the audio file: {e}")

        return tensor(speech_array, dtype=float32), tensor(arr, dtype=long)

    def collate_fn(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sequences, labels = zip(*batch)
        sequences_padded = pad_sequence(sequences, batch_first=True)
        sequences_labels = pad_sequence(labels, batch_first=True)
        return sequences_padded, sequences_labels


class IemocapDataset(Dataset):
    def __init__(self, name, split="train"):
        # with open(list_path, "r") as f:
        #     lines = f.readlines()

        data = torch.load(f'data/iemocap_{name}_{split}.pt')
        self.data = data["embeddings"]
        self.labels = data["labels"]

        if split == "train":
            torch.manual_seed(42)  # 为了可重复性，设置随机种子
            indices = torch.randperm(len(self.data))
            self.data = self.data[indices]
            self.labels = self.labels[indices]

            self.data = self.data[:128]
            self.labels = self.labels[:128]
        # for line in lines:
        #     wav_path, speaker_id = line.split()
        #     self.data.append(wav_path)
        #     self.labels.append(int(speaker_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = self.data[index]
        label = self.labels[index]

        # label = torch.zeros(self.get_classes())
        #
        # wav, sr = torchaudio.load(wav_path)
        # # feat = self.data[index]
        # label[speaker_id] = 1
        #
        # wav = wav.squeeze(0)
        # if wav.size(0) > 80000:
        #     wav = wav[:80000]
        return feature, label

    def get_classes(self):
        return int(int(max(self.labels))+1)


class MascDataset(Dataset):
    def __init__(self, name, split="train", n_fold=0):
        data = torch.load(f'data/masc_{name}_{split}_{n_fold}.pt')
        self.data = data["embeddings"]
        # self.labels = torch.argmax(data["labels"], dim=1)
        self.labels = data["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = self.data[index]
        label = self.labels[index]

        return feature, label

