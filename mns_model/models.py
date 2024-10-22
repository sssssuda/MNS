import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    level=logging.INFO)


class DNN(nn.Module):
    def __init__(self, input_shape=26, num_classes=18):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()

        self.layer1 = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )

        self.layer7 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )

        self.layer8 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

        self.feature = nn.ReLU()  # Named "feature1" in Keras
        self.output_layer = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer7(x)
        x = self.layer8(x)
        feature1 = self.feature(x)  # Equivalent to "feature1" layer in Keras
        out = self.output_layer(feature1)
        out = self.softmax(out)
        return out, feature1


class CNN(nn.Module):
    def __init__(self, input_channels, input_length, num_classes=18):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_pooling = self._global_stat_pooling
        self.fc1 = nn.Linear(128 * 2, 128)

        self.fc2 = nn.Linear(128, num_classes)

    def _global_stat_pooling(self, x):
        mean = torch.mean(x, dim=2)  # 计算平均值
        std = torch.std(x, dim=2)  # 计算方差
        pooled = torch.cat((mean, std), dim=1)  # 将均值和方差拼接
        return pooled

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pooling(x)
        feat = self.fc1(x)

        x = self.fc2(feat)

        return x, feat


class SeperateModel(nn.Module):
    def __init__(self, input_shape, num_classes=18):
        super(SeperateModel, self).__init__()
        self.input_shape = input_shape
        input_length, input_channels = input_shape

        self.dnn = DNN(input_shape=input_length, num_classes=num_classes)  # Flatten 在 DNN 中处理
        parameters = sum([torch.numel(params) for params in self.dnn.parameters()])
        logging.info("model params: {}".format(parameters))

        self.cnn = CNN(input_channels=input_channels, input_length=input_length, num_classes=num_classes)
        parameters = sum([torch.numel(params) for params in self.cnn.parameters()])
        logging.info("model params: {}".format(parameters))

    def forward(self, x):
        dnn_out, feature1 = self.dnn(x)
        cnn_out, feature2 = self.cnn(x)

        return dnn_out, cnn_out


class FusionModel(nn.Module):
    def __init__(self, input_shape, mode="train", num_classes=18):
        super(FusionModel, self).__init__()
        self.input_shape = input_shape
        # 假设 input_shape 为 (length, channels)
        input_length, input_channels = input_shape

        self.dnn = DNN(input_shape=input_length, num_classes=num_classes)  # Flatten 在 DNN 中处理
        parameters = sum([torch.numel(params) for params in self.dnn.parameters()])
        logging.info("model params: {}".format(parameters))
        if mode == "train":
            self.dnn.load_state_dict(torch.load("model/dnn_model_best.pth"))

        self.cnn = CNN(input_channels=input_channels, input_length=input_length, num_classes=num_classes)
        parameters = sum([torch.numel(params) for params in self.cnn.parameters()])
        logging.info("model params: {}".format(parameters))
        if mode == "train":
            self.cnn.load_state_dict(torch.load("model/cnn_model_best.pth"))

            logging.info("Load cnn and dnn model from 'model/cnn_model_5dnn_best.pth'"
                         " and 'model/dnn_model_5dnn_best.pth'")

        # 融合后的全连接层
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        dnn_out, feature1 = self.dnn(x)
        cnn_out, feature2 = self.cnn(x)

        # 融合特征
        fusion_feature = torch.cat((feature1, feature2), dim=1)

        # 输出层
        output = self.fusion_layer(fusion_feature)

        # output = (dnn_out + cnn_out) / 2
        return output, feature1, feature2


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)

    def forward(self, x, v):
        # x (batch,frames,dim)
        # v (batch,frames,classes)
        q = self.Wq(x)
        k = self.Wk(x)

        attention_scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1)), dim=-1)
        attention_output = torch.matmul(attention_scores, v)

        return attention_output  # , attention_scores


class DNNClassifier(nn.Module):

    def __init__(self, input_size, num_classes, dropout_rate=0.3):
        super(DNNClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),

            nn.Linear(128, num_classes),  # 输出层根据分类任务设置 num_classes
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.model(x)

        return x

