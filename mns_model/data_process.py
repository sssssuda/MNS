import math
import pickle
import numpy as np
import python_speech_features as psf
import torch

from mns_model.utils import load_masc_data
import soundfile as sf


def load_data_frompkl(path=None, split="train"):
    # masc
    # data: (utterance, length)
    # label: (utterance, 18) one-hot编码
    with open(f"data/{split}_data.pkl", "rb") as f:
        data = pickle.load(f)

    with open(f"data/{split}_label.pkl", "rb") as f:
        label = pickle.load(f)

    return data, label


def load_data_fromdat(path=None, split="train"):
    # iemocap, frame(26)
    with open(f"data/iemocap_{split}_data.pkl", "rb") as f:
        data = pickle.load(f)

    with open(f"data/iemocap_{split}_label.pkl", "rb") as f:
        label = pickle.load(f)

    return data, label


def load_data_fromlist(path=None, split="train", n_fold=0):
    datas, labels = load_masc_data(split=split, n_fold=n_fold)
    data, label = [], []
    for wav_path, idx_label in zip(datas, labels):
        wav, sr = sf.read(wav_path)
        data.append(wav)
        oh_label = np.zeros(18)
        oh_label[idx_label] = 1
        label.append(oh_label)

    return data, label


def get_LogFillterBank(x, sample_rate=8000, nfilt=26):
    """
    将信号变为log fillterbank特征 (使用librosa提取log-mel谱图)
    :param x: signal
    :param sample_rate: 默认16000
    :return: 特征值 (utterance, frames, 26)
    """
    fillterBank = []
    for i in range(len(x)):
        fb = psf.logfbank(x[i], samplerate=sample_rate, nfilt=nfilt).astype("float32")
        fillterBank.append(fb)
    # fillterBank = np.array(fillterBank)
    return fillterBank


def enframe(data, label):
    """
    对话语集特征做分帧处理,得到帧级特征
    :param data: 话语级数据
    :param label: 说话人标签
    :return: 帧级数据、标签(每句话中N帧)
    """
    frame_data = []
    frame_labels = []
    for i in range(len(data)):
        utterance = data[i]  # (T, F)
        frame_data.append(utterance)
        # 假设 label 是 one-hot 编码，转换为类别索引
        frame_label = np.argmax(label[i])
        frame_labels.extend([frame_label] * utterance.shape[0])

    frame_data = np.vstack(frame_data)  # (N*T, F)
    frame_labels = np.array(frame_labels)  # (N*T,)

    return frame_data, frame_labels


def get_segments(lfb, labels):
    _max = 20
    labels_s = []
    data_s = []
    for i in range(len(lfb)):
        utt = lfb[i]
        segments = math.floor(len(utt) / _max)
        length = len(utt)
        if segments <= 0:
            if _max - length <= length:
                labels_s.append(np.argmax(labels[i]))
                new_utt = np.concatenate((utt, utt[:_max - length]), axis=0)
                data_s.append(new_utt)

        else:
            rest = length - segments * _max
            r_u = utt[segments * _max:]
            for k in range(segments):
                labels_s.append(np.argmax(labels[i]))
                new_utt = utt[k * _max: k * _max + _max]
                data_s.append(new_utt)
            if rest > 0:
                if rest >= _max / 2:
                    labels_s.append(np.argmax(labels[i]))
                    new_utt = np.concatenate((r_u, r_u[:_max - rest]), axis=0)
                    data_s.append(new_utt)

    return np.array(data_s), np.array(labels_s)


def get_segments_utterance(lfb, labels, lstm=True):
    _max = 20
    if lstm:
        expected_shape = (20, 26)
    else:
        expected_shape = (20, 26, 1)
    labels_s = []
    data_s = []
    for i in range(len(lfb)):
        utt = lfb[i]
        utt_data, utt_label = [], []
        segments = math.floor(len(utt) / _max)
        length = len(utt)
        if segments <= 0:
            if _max - length <= length:
                utt_label.append(labels[i])
                new_utt = np.concatenate((utt, utt[:_max - length]), axis=0)
                utt_data.append(new_utt.reshape(expected_shape))

        else:
            rest = length - segments * _max
            r_u = utt[segments * _max:]
            for k in range(segments):
                utt_label.append(labels[i])
                new_utt = utt[k * _max: k * _max + _max]
                utt_data.append(new_utt.reshape(expected_shape))
            if rest > 0:
                if rest >= _max / 2:
                    utt_label.append(labels[i])
                    new_utt = np.concatenate((r_u, r_u[:_max - rest]), axis=0)
                    utt_data.append(new_utt.reshape(expected_shape))

        data_s.append(np.array(utt_data))
        labels_s.append(labels[i])

    return data_s, labels_s


def get_speaker_embedding(model, name="iemocap", nfilt=26):
    model.eval()

    if name == "iemocap":
        data, labels = load_data_fromdat(split="train")
    else:
        data, labels = load_data_frompkl(split="train")

    lfb = get_LogFillterBank(data, sample_rate=16000, nfilt=nfilt)

    emb = {}
    for i in range(len(lfb)):
        utt = torch.tensor(lfb[i]).unsqueeze(0)  # Prepare input for the model
        label = labels[i]

        # Forward pass through the model to get the embeddings
        with torch.no_grad():
            _, embedding = model(utt)  # Extract the second output which is the embedding

        # Convert embedding to numpy array
        embedding = embedding.squeeze(0).cpu().numpy()

        # If the label (speaker) already exists, append to its list
        if label in emb:
            emb[label].append(embedding)
        else:
            emb[label] = [embedding]

    # Average the embeddings for each speaker and save them locally
    averaged_emb = {speaker: np.mean(np.stack(embeddings), axis=0) for speaker, embeddings in emb.items()}

    torch.save(averaged_emb, "speaker_embeddings.pt")

    return averaged_emb


def duration():
    data, _ = load_data_fromdat(split="train")
    print("iemocap-train: {}h".format(len(data) * 0.015 / 60 / 60))
    data, _ = load_data_fromdat(split="test")
    print("iemocap-test: {}h".format(len(data) * 0.015 / 60 / 60))

    data, _ = load_data_frompkl(split="train")
    wav = 0
    for i in range(len(data)):
        wav += len(data[i])
    print("masc-train: {}h".format(wav / 8000 / 60 / 60))

    data, _ = load_data_frompkl(split="test")
    wav = 0
    for i in range(len(data)):
        wav += len(data[i])
    print("masc-train: {}h".format(wav / 8000 / 60 / 60))


