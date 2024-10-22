import logging

import torchaudio
from sklearn.metrics import accuracy_score,roc_curve
import numpy as np
import torch
import random

from torch import nn

from mns_model.models import CNN, DNN

MODELCLASS = [CNN, DNN]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def cosine(xi, mean):
    x_mean_norm = np.linalg.norm(mean)
    x_i_norms = np.linalg.norm(xi, axis=1)

    cosine_similarity = np.dot(xi, mean) / (x_i_norms * x_mean_norm)

    return cosine_similarity


def compute_metrics(labels, predicts, answers, scores):
    """
    计算模型的评价指标
    :param labels: 训练集的真实标签
    :param predicts: 训练集的预测标签
    :param answers: 训练数据对每一类的值(one-hot值)
    :param scores: 训练数据对每一类的预测值(softmax输出)
    :return: 准确率、等错误率、dcf
    """
    acc = accuracy_score(labels, predicts)
    fpr, tpr, thresholds = roc_curve(answers, scores)

    far, tar = fpr, tpr
    frr = 1 - tar
    cr = 10
    ca = 1
    pt = 0.01

    min_dirr = min([abs(far[i] - frr[i]) for i in range(len(far))])

    for i in range(len(far)):
        if abs(far[i] - frr[i]) == min_dirr:
            eer = (far[i] + frr[i]) / 2
            dcf = cr * frr[i] * pt + ca * far[i] * (1 - pt)
            break

    return acc, eer, dcf


def model_output(test_lfb, test_label, model, lstm=False):
    model.eval()
    model.to("cuda")

    y_preds = []
    fbanks = []
    score_labels = []
    true_labels = []
    vote_labels = []
    scores = []
    answers = []
    for i in range(len(test_lfb)):
        answer = test_label[i]
        utterance = np.array(test_lfb[i])

        # 可视化FBank
        mean_feature = np.mean(utterance, axis=0)
        std_feature = np.std(utterance, axis=0)
        fbanks.append(np.concatenate([mean_feature, std_feature]))

        if lstm:
            utterance = torch.from_numpy(utterance).to("cuda")
        else:
            utterance = torch.from_numpy(utterance).unsqueeze(2).to("cuda")
        utterance_label = np.squeeze(answer)  # (18, )
        if any(isinstance(model, model_class) for model_class in MODELCLASS):
            utterance_predict, _ = model(utterance)
        else:
            utterance_predict, _, _ = model(utterance)  # (frames, 18)

        utterance_predict = utterance_predict.detach().cpu().numpy()

        mean_predict = np.mean(utterance_predict, axis=0)  # (18, )

        # 一句话中每一帧的预测类标, 类标最多的为该句的类标
        frame_index = np.argmax(utterance_predict, axis=1)  # (frames, )
        vote_predict = np.argmax(np.bincount(frame_index))

        # 一句话中每一帧对应每一类的得分平均, 得分最高的类为该句的类标
        score_predict = np.argmax(mean_predict)
        y_preds.append(mean_predict)

        # 实际类标
        true_label = np.argmax(utterance_label)


        vote_labels.append(vote_predict)
        score_labels.append(score_predict)
        true_labels.append(true_label)
        scores.append(mean_predict.tolist())
        answers.append(answer.tolist())

    vote_labels = np.array(vote_labels)
    score_labels = np.array(score_labels)
    true_labels = np.array(true_labels)
    scores = np.array(scores).reshape(-1, 1)
    answers = np.array(answers).reshape(-1, 1)

    # vote_labels 可以替换为score_labels,vote是每一帧的投票结果最大值，score是每一帧预测概率平均值最大值
    return true_labels, vote_labels, answers, scores


def model_cosine_output0(x_test, x_mean, test_label, model):
    """
       计算对比算法的预测标签、对每一类的预测值
       :param x_mean: 每个预测说话人的d-vector特征表示
       :param x_test:  每句话语级的d-vector待测试表示
       :return: 预测标签、对每一类的预测值
       """
    model.to("cuda")

    true_labels = []
    answers = []
    scores = []
    for i in range(len(x_test)):
        answer = test_label[i]
        answers.append(answer.tolist())
        utterance_label = np.squeeze(answer)  # (18, )
        true_label = np.argmax(utterance_label)
        true_labels.append(true_label)
        for j in range(len(x_mean)):
            x_i = torch.from_numpy(np.array(x_test[i])).unsqueeze(2).to("cuda")
            with torch.no_grad():
                pred_out, dnn_out, cnn_out = model(x_i)
            x_i = ((dnn_out + cnn_out) / 2).detach().cpu().numpy()
            x_i = np.mean(x_i, axis=0)
            score = 1 - cosine(x_i, x_mean[j])
            scores.append(score)
    scores = np.array(scores)

    all_scores = scores.reshape(-1, 1)
    speaker_scores = scores.reshape(-1, 18)
    cosine_labels = np.argmax(speaker_scores, axis=1)  # (utterances, )

    true_labels = np.array(true_labels)
    answers = np.array(answers).reshape(-1, 1)

    return true_labels, cosine_labels, answers, all_scores


def model_cosine_output(x_test, test_label, model):
    model.to("cuda")

    y_preds = []
    true_labels = []
    answers = []
    for i in range(len(x_test)):
        x_mean = np.mean(x_test[i], axis=0)
        x_i = np.array(x_test[i])
        with torch.no_grad():
            pred_out, _, _ = model(torch.from_numpy(x_i).unsqueeze(2).to("cuda"))
        w_i = 1 - cosine(x_i, x_mean)
        y_pred = np.sum(w_i[:, np.newaxis] * pred_out.detach().cpu().numpy(), axis=0) / len(x_test[i])
        y_preds.append(y_pred)

        answer = test_label[i]
        answers.append(answer.tolist())
        utterance_label = np.squeeze(answer)  # (18, )
        true_label = np.argmax(utterance_label)
        true_labels.append(true_label)

    y_preds = np.array(y_preds)
    cosine_labels = np.argmax(y_preds, axis=1)

    true_labels = np.array(true_labels)
    answers = np.array(answers).reshape(-1, 1)
    all_scores = y_preds.reshape(-1, 1)

    return true_labels, cosine_labels, answers, all_scores


def get_means(train_lfb, train_label, model):
    # train_lfb: (utterance, frames, 26)
    # train_label: (utterance, 18)

    model.to("cuda")
    train_vectors = [[] for _ in range(18)]
    for i in range(len(train_lfb)):
        true_label = np.argmax(np.squeeze(train_label[i]))
        utterance = torch.from_numpy(np.array(train_lfb[i])).unsqueeze(2).to("cuda")
        with torch.no_grad():
            _, dnn_out, cnn_out = model(utterance)
            utterance_predict = (dnn_out + cnn_out) / 2

        mean_predict = np.mean(utterance_predict.detach().cpu().numpy(), axis=0)
        train_vectors[true_label].append(mean_predict.tolist())

    train_vectors = np.array(train_vectors)  # (18, utterances, 128)
    train_vectors = np.mean(train_vectors, axis=1)  # (18, 128)

    return train_vectors


def model_attention_output(x_test, test_label, model, attention_layer):
    model.to("cuda")
    attention_layer.to("cuda")

    y_preds = []
    true_labels = []
    answers = []
    attention_labels = []

    for i in range(len(x_test)):
        x_i = torch.from_numpy(np.array(x_test[i])).unsqueeze(2).to("cuda")
        x_att = torch.from_numpy(np.array(x_test[i])).unsqueeze(0).to("cuda")
        with torch.no_grad():
            pred_out, _, _ = model(x_i)
            attention_output = attention_layer(x_att, pred_out.unsqueeze(0)).squeeze(0)  # shape: (frames, classes)

        # Weight the model's output with attention weights
        weighted_output = torch.sum(attention_output, dim=0)
        weighted_output = torch.softmax(weighted_output, dim=-1).detach().cpu().numpy()

        y_preds.append(weighted_output)
        attention_labels.append(np.argmax(weighted_output))

        # Collect true labels
        answer = test_label[i]
        answers.append(answer.tolist())
        utterance_label = np.squeeze(answer)  # (18, )
        true_label = np.argmax(utterance_label)
        true_labels.append(true_label)

    y_preds = np.array(y_preds)
    attention_labels = np.argmax(y_preds, axis=1)

    true_labels = np.array(true_labels)
    answers = np.array(answers).reshape(-1, 1)
    all_scores = y_preds.reshape(-1, 1)

    return true_labels, attention_labels, answers, all_scores


def train_attention_model(trainloader, model, attention_layer, epochs=10, lr=0.0001):
    optimizer = torch.optim.Adam(attention_layer.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    model.to("cuda")
    attention_layer.train()
    attention_layer.to("cuda")

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        total_loss, accuracy = 0, 0
        for x_train, train_label in trainloader:
            # x_train (1, frames, dim)
            x_i = torch.from_numpy(np.array(x_train)).squeeze(0).unsqueeze(2).to("cuda")  # (frames, dim, 1)
            x_att = torch.from_numpy(np.array(x_train)).to("cuda")  # (1, frames, dim)
            true_label = torch.tensor(train_label).to("cuda")  # (1, classes)   one-hot
            optimizer.zero_grad()

            pred_out, _, _ = model(x_i)   # (frames, classes)
            attention_out = attention_layer(x_att, pred_out.unsqueeze(0)).squeeze(0)     # (frames, classes)
            weighted_output = torch.sum(attention_out, dim=0, keepdim=True)    # (1, classes)
            weighted_output = torch.softmax(weighted_output, dim=-1)

            x_label = torch.argmax(weighted_output, dim=1)
            y_label = torch.argmax(true_label, dim=1)
            if x_label == y_label:
                accuracy += 1

            loss = loss_fn(weighted_output.squeeze(0), true_label.squeeze(0))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logging.info(f"Train Loss: {total_loss / len(trainloader):.4f}, Train Accuracy: {accuracy / len(trainloader):.4f}")

    return attention_layer


def load_iemocap_data(list_path, split="train"):
    data = []
    labels = []

    # Read the list of files and speaker IDs
    with open(list_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        wav_path, speaker_id = line.split()
        data.append(wav_path)
        labels.append(int(speaker_id))

    return data, labels


def load_masc_data(split, n_fold):
    data = []
    labels = []

    with open(f"data/masc_{split}_fold_{n_fold}.list") as f:
        lines = f.readlines()

    for line in lines:
        wav_path, speaker_id = line.split()
        data.append(wav_path)
        labels.append(int(speaker_id))

    return data, labels




